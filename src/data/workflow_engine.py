import os
import logging
import tempfile
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import threading
import collections

# Assuming these imports are correct relative to your project structure
from .workflow import Workflow, WorkflowStep
from .execution_history import ExecutionRecord, history_manager
from src.tools.actions.action_registry import registry as action_registry
from src.tools.sources.source_registry import registry as source_registry
from src.tools.actions.waifuc_actions import WaifucActionWrapper

from waifuc.source import LocalSource
from waifuc.action import TerminalAction
from waifuc.export import SaveExporter, TextualInversionExporter

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# --- NEW: Define all actions that generate or modify tags ---
# Based on the tagging.py file you provided.
TAG_RELATED_ACTIONS = {
    'TaggingAction',
    'TagFilterAction',
    'TagOverlapDropAction',
    'TagDropAction',
    'BlacklistedTagDropAction',
    'TagRemoveUnderlineAction',
    'TagAppendAction',
}

class QueuedTask:
    def __init__(self,
                 execution_record: ExecutionRecord,
                 workflow: Workflow,
                 source_type: str,
                 source_params: Dict[str, Any],
                 output_directory: str,
                 progress_callback: Optional[Callable[[str, float, str], None]] = None,
                 cancel_event: threading.Event = None):
        self.execution_record = execution_record
        self.workflow = workflow
        self.source_type = source_type
        self.source_params = source_params
        self.output_directory = output_directory
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event

class WorkflowEngine:
    def __init__(self, max_workers: int = 1):
        if max_workers != 1:
            logger.warning(f"WorkflowEngine with queuing enforces max_workers=1. Provided value {max_workers} was overridden.")
        self.max_workers = 1
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running_tasks: Dict[str, Tuple[Any, ExecutionRecord, threading.Event]] = {}
        self._task_queue = collections.deque()
        self._queue_lock = threading.Lock()
        self._current_processing_record_id: Optional[str] = None
        os.makedirs("logs", exist_ok=True)

    def execute_workflow(self, workflow: Workflow,
                       source_type: str, source_params: Dict[str, Any],
                       output_directory: str,
                       progress_callback: Optional[Callable[[str, float, str], None]] = None) -> ExecutionRecord:
        record = history_manager.create_record(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            source_type=source_type,
            source_params=source_params,
            output_directory=output_directory
        )
        record.status = "queued"
        history_manager.save_record(record)
        logger.info(f"Workflow '{workflow.name}' (Record ID: {record.id}) enqueued.")
        cancel_event = threading.Event()
        queued_item = QueuedTask(
            execution_record=record,
            workflow=workflow,
            source_type=source_type,
            source_params=source_params,
            output_directory=output_directory,
            progress_callback=progress_callback,
            cancel_event=cancel_event
        )
        with self._queue_lock:
            self._task_queue.append(queued_item)
            logger.debug(f"Record ID: {record.id} added to queue. Queue size: {len(self._task_queue)}")
        self._try_process_next_task_from_queue()
        return record

    def _try_process_next_task_from_queue(self):
        with self._queue_lock:
            if self._current_processing_record_id is not None:
                logger.debug(f"Engine busy with Record ID: {self._current_processing_record_id}. Queue will wait.")
                return
            if not self._task_queue:
                logger.debug("Task queue is empty. Nothing to process.")
                return
            next_queued_item: QueuedTask = self._task_queue.popleft()
            record_to_process = next_queued_item.execution_record
            self._current_processing_record_id = record_to_process.id
            record_to_process.status = "processing_setup"
            history_manager.save_record(record_to_process)
            logger.info(f"Dequeued Record ID: {record_to_process.id} for execution.")

        cancel_event_for_task = next_queued_item.cancel_event
        self._running_tasks[record_to_process.id] = (None, record_to_process, cancel_event_for_task)
        future = self.executor.submit(
            self._execute_workflow_internal,
            workflow=next_queued_item.workflow,
            source_type=next_queued_item.source_type,
            source_params=next_queued_item.source_params,
            output_directory=next_queued_item.output_directory,
            record=record_to_process,
            progress_callback=next_queued_item.progress_callback,
            cancel_event=cancel_event_for_task
        )
        self._running_tasks[record_to_process.id] = (future, record_to_process, cancel_event_for_task)
        future.add_done_callback(lambda f, rid=record_to_process.id: self._on_task_done(f, rid))

    def _on_task_done(self, future_object, record_id: str):
        try:
            exception = future_object.exception()
            if exception:
                logger.error(f"Task (Record ID: {record_id}) execution raised an exception in future: {exception}", exc_info=exception)
                if record_id in self._running_tasks:
                    _, record, _ = self._running_tasks[record_id]
                    if record.status not in TERMINAL_STATUSES:
                        record.fail(f"Future exception: {str(exception)}")
                        history_manager.save_record(record)
        finally:
            if record_id in self._running_tasks:
                del self._running_tasks[record_id]
                logger.debug(f"Removed Record ID {record_id} from _running_tasks via _on_task_done callback.")
            with self._queue_lock:
                if self._current_processing_record_id == record_id:
                    self._current_processing_record_id = None
                    logger.debug(f"Engine now idle (was processing Record ID: {record_id}).")
                else:
                    logger.warning(f"Mismatch on task completion: _current_processing_record_id was {self._current_processing_record_id}, completed task was {record_id}.")
                    if self._current_processing_record_id is not None and not self._running_tasks.get(self._current_processing_record_id) :
                         self._current_processing_record_id = None
            logger.debug(f"Triggering next task processing after Record ID {record_id} completion.")
            self._try_process_next_task_from_queue()

    def _execute_workflow_internal(self, workflow: Workflow,
                                  source_type: str, source_params: Dict[str, Any],
                                  output_directory: str, record: ExecutionRecord,
                                  progress_callback: Optional[Callable[[str, float, str], None]] = None,
                                  cancel_event: Optional[threading.Event] = None) -> None:
        class CancelledError(Exception):
            pass

        task_logger = logging.getLogger(f"workflow.{record.id}")
        file_handler = None
        if not task_logger.handlers:
            log_file = os.path.join("logs", f"{record.id}_log.txt")
            file_handler = logging.FileHandler(log_file, 'w', 'utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            task_logger.addHandler(file_handler)
            task_logger.setLevel(logging.INFO)
        else:
            file_handler = task_logger.handlers[0] if task_logger.handlers else None

        temp_dir = None
        try:
            if record.status != "processing" :
                record.status = "processing"
                if record.start_time is None:
                     record.start_time = time.time()
                history_manager.save_record(record)

            os.makedirs(output_directory, exist_ok=True)
            temp_dir = tempfile.mkdtemp()
            temp_input_dir = os.path.join(temp_dir, 'input')
            os.makedirs(temp_input_dir, exist_ok=True)

            task_logger.info(f"Task {record.id} started. Workflow: {workflow.name}, Source: {source_type}, Output: {output_directory}")
            if progress_callback:
                progress_callback("获取图像", 0.0, "准备图像来源...")
            if cancel_event and cancel_event.is_set():
                raise CancelledError("任务在获取图像前被取消")

            input_dir_for_processing = ""
            try:
                source = source_registry.create_source(source_type, **source_params)
                record.add_step_log("source_preparation", source_type, "started", "创建图像来源")
                task_logger.info("从来源获取图像...")
                if progress_callback: progress_callback("获取图像", 0.1, "正在获取图像...")
                if source_type == "LocalSource":
                    input_dir_for_processing = source_params.get("directory", "")
                    if not os.path.exists(input_dir_for_processing):
                        raise FileNotFoundError(f"输入目录不存在: {input_dir_for_processing}")
                    total_files = sum(1 for f in os.listdir(input_dir_for_processing)
                                    if os.path.isfile(os.path.join(input_dir_for_processing, f)) and
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                    record.total_images = total_files
                    task_logger.info(f"发现 {total_files} 个图像文件于 {input_dir_for_processing}")
                else:
                    task_logger.info("开始下载图像...")
                    if progress_callback: progress_callback("获取图像", 0.2, "下载图像...")
                    source.source.export(SaveExporter(temp_input_dir, no_meta=False))
                    total_files = sum(1 for f in os.listdir(temp_input_dir)
                                    if os.path.isfile(os.path.join(temp_input_dir, f)) and
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                    record.total_images = total_files
                    task_logger.info(f"已下载 {total_files} 个图像文件到 {temp_input_dir}")
                    input_dir_for_processing = temp_input_dir
                record.add_step_log("source_preparation", source_type, "completed", f"成功获取 {record.total_images} 个图像文件")
            except Exception as e:
                error_msg = f"获取图像失败: {str(e)}"
                task_logger.error(error_msg, exc_info=True)
                record.add_step_log("source_preparation", source_type, "failed", error_msg)
                record.fail(error_msg)
                history_manager.save_record(record)
                if progress_callback: progress_callback("错误", 0.0, error_msg)
                return
            
            if cancel_event and cancel_event.is_set(): raise CancelledError("任务在处理步骤前被取消")
            current_dir_for_steps = input_dir_for_processing
            
            # --- INTELLIGENT WORKFLOW LOGIC ---
            # 1. Determine the workflow's intent based on action names
            is_tagging_workflow = any(step.action_name in TAG_RELATED_ACTIONS for step in workflow.steps)
            if is_tagging_workflow:
                task_logger.info("Tagging-related action detected in workflow. Final export will be checked for tags.")
            else:
                task_logger.info("No tagging actions in workflow. Final export will only save images.")

            for i, step in enumerate(workflow.steps):
                step_progress_base = 0.3 + (i / len(workflow.steps)) * 0.6
                
                step_output_dir = os.path.join(temp_dir, f"step_{i+1}_{uuid.uuid4().hex[:8]}")
                os.makedirs(step_output_dir, exist_ok=True)

                task_logger.info(f"Executing step {i+1}/{len(workflow.steps)}: {step.action_name} (In: {current_dir_for_steps}, Out: {step_output_dir})")
                record.add_step_log(step.id, step.action_name, "started", f"Starting step {i+1}/{len(workflow.steps)}")
                if progress_callback: progress_callback("Processing images", step_progress_base, f"Executing step {i+1}/{len(workflow.steps)}: {step.action_name}")
                if cancel_event and cancel_event.is_set(): raise CancelledError(f"Task cancelled before executing step {step.action_name}")
                try:
                    action = action_registry.create_action(step.action_name, **step.params)
                    action_instance = action.action if isinstance(action, WaifucActionWrapper) and hasattr(action, 'action') else action
                    
                    if isinstance(action_instance, TerminalAction):
                        action_instance.output_directory = output_directory
                        task_logger.info(f"Step {i+1} is a TerminalAction, output directory injected: {output_directory}")
                    
                    source_for_step = LocalSource(current_dir_for_steps)
                    processed_output = source_for_step.attach(action_instance)
                    
                    # For ALL steps, use SaveExporter to preserve the metadata chain.
                    # The final conversion to .txt or simple image save is handled AFTER the loop.
                    processed_output.export(SaveExporter(step_output_dir, no_meta=False))
                    current_dir_for_steps = step_output_dir # The output of this step is the input for the next.
                    
                    record.add_step_log(step.id, step.action_name, "completed", f"Step {i+1}/{len(workflow.steps)} completed successfully.")
                    if progress_callback: progress_callback("Processing images", step_progress_base + 0.6/len(workflow.steps), f"Step {i+1}/{len(workflow.steps)} complete")

                except Exception as e_step:
                    error_msg_step = f"Step {i+1} ({step.action_name}) failed: {str(e_step)}"
                    task_logger.error(error_msg_step, exc_info=True)
                    record.add_step_log(step.id, step.action_name, "failed", error_msg_step)
                    record.fail(f"Workflow aborted due to failure in step {step.action_name}: {error_msg_step}")
                    history_manager.save_record(record)
                    if progress_callback: progress_callback("Error", step_progress_base, f"Step {step.action_name} failed, workflow aborted.")
                    return
            
            if cancel_event and cancel_event.is_set(): raise CancelledError("Task cancelled before finalizing output.")

            # --- FINAL EXPORT LOGIC (REVISED) ---
            # After all steps are complete, `current_dir_for_steps` holds the result.
            # Now, decide how to export it to the final `output_directory`.
            task_logger.info(f"Finalizing export from last step's directory: {current_dir_for_steps}")
            final_source = LocalSource(current_dir_for_steps)
            final_exporter = None

            if is_tagging_workflow:
                # The workflow was *intended* for tagging. Now, VERIFY if tags actually exist.
                has_actual_tags = False
                try:
                    # Peek at the first item to see if it has a non-empty tags dictionary.
                    first_item = next(iter(final_source))
                    if first_item.meta.get('tags'):
                        has_actual_tags = True
                except StopIteration:
                    # The source is empty, so no items and no tags.
                    has_actual_tags = False
                
                if has_actual_tags:
                    # If tags were found, use the exporter that creates .txt files.
                    task_logger.info("Verified that tags exist. Exporting final result with TextualInversionExporter.")
                    final_exporter = TextualInversionExporter(output_directory)
                else:
                    # If no tags were found despite the intent, just save the images.
                    task_logger.info("Workflow was intended for tagging, but no actual tags were found. Exporting images only.")
                    final_exporter = SaveExporter(output_directory, no_meta=True)
            else:
                # If it was never a tagging workflow, just save the images.
                task_logger.info("Non-tagging workflow. Exporting final result with SaveExporter (no_meta=True).")
                final_exporter = SaveExporter(output_directory, no_meta=True)
            
            # Perform the final, decisive export.
            final_source.export(final_exporter)
            
            # Count the final files in the output directory
            final_output_files_count = 0
            if os.path.exists(output_directory):
                final_output_files_count = len([f for f in os.listdir(output_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])
            
            task_logger.info(f"Saved {final_output_files_count} files to the final output directory: {output_directory}")
            
            record.complete(
                total_images=record.total_images if record.total_images is not None else 0,
                processed_images=final_output_files_count,
                success_images=final_output_files_count,
                failed_images=0
            )
            history_manager.save_record(record)
            completion_log_message = getattr(record, 'message', record.status)
            task_logger.info(f"Workflow execution complete. Record ID: {record.id}. Status: {record.status}, Details: {completion_log_message}")
            if progress_callback: progress_callback("Complete", 1.0, f"Processing complete. Total images: {record.total_images}, Successful: {final_output_files_count}")

        except CancelledError as e:
            error_msg = str(e)
            task_logger.info(f"Record ID {record.id}: {error_msg} (Caught in _execute_workflow_internal)")
            record.fail(error_msg)
            history_manager.save_record(record)
            if progress_callback:
                progress_val = 0.0
                progress_callback("Cancelled", progress_val, error_msg)
        
        except Exception as e:
            error_msg = f"An unexpected error occurred during workflow execution: {str(e)}"
            task_logger.error(f"Record ID {record.id}: {error_msg}", exc_info=True)
            if record.status not in TERMINAL_STATUSES:
                record.fail(error_msg)
                history_manager.save_record(record)
            if progress_callback:
                progress_callback("Error", 0.0, error_msg)
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    task_logger.info(f"Temporary directory {temp_dir} removed for Record ID {record.id}.")
                except Exception as e_rm_temp: task_logger.error(f"Failed to remove temporary directory {temp_dir} for Record ID {record.id}: {e_rm_temp}")
            if file_handler and task_logger:
                task_logger.removeHandler(file_handler)
                file_handler.close()

    def get_running_tasks(self) -> Dict[str, ExecutionRecord]:
        running_records = {}
        for task_id, (_future, record, _cancel_event) in list(self._running_tasks.items()):
            if _future and not _future.done() and record.status not in TERMINAL_STATUSES:
                running_records[task_id] = record
        return running_records

    def get_queued_tasks_info(self) -> List[Dict[str, Any]]:
        with self._queue_lock:
            return [
                {
                    "record_id": qt.execution_record.id,
                    "workflow_name": qt.execution_record.workflow_name,
                    "status": qt.execution_record.status,
                    "queued_time": qt.execution_record.create_time.isoformat() if qt.execution_record.create_time else None
                } for qt in self._task_queue
            ]

    def get_current_processing_task_info(self) -> Optional[Dict[str, Any]]:
        with self._queue_lock: record_id = self._current_processing_record_id
        if record_id and record_id in self._running_tasks:
            _future, record, _ = self._running_tasks[record_id]
            return {
                "record_id": record.id, "workflow_name": record.workflow_name, "status": record.status,
                "start_time": record.start_time.isoformat() if hasattr(record.start_time, 'isoformat') else str(record.start_time),
            }
        return None

    def cancel_task(self, execution_record_id: str) -> bool:
        logger.info(f"Attempting to cancel task with Record ID: {execution_record_id}")
        if execution_record_id in self._running_tasks:
            _future, record, cancel_event_to_set = self._running_tasks[execution_record_id]
            if record.status not in TERMINAL_STATUSES:
                logger.info(f"Signaling cancellation for actively processing Record ID: {execution_record_id}")
                cancel_event_to_set.set()
                return True
            else:
                logger.info(f"Record ID: {execution_record_id} is already finished (status: {record.status}), cannot cancel.")
                return False
        with self._queue_lock:
            task_to_remove_from_queue: Optional[QueuedTask] = None
            for queued_item_in_q in self._task_queue:
                if queued_item_in_q.execution_record.id == execution_record_id:
                    task_to_remove_from_queue = queued_item_in_q; break
            if task_to_remove_from_queue:
                self._task_queue.remove(task_to_remove_from_queue)
                record_to_cancel = task_to_remove_from_queue.execution_record
                record_to_cancel.fail("Task has been cancelled from the queue")
                history_manager.save_record(record_to_cancel)
                logger.info(f"Record ID: {execution_record_id} removed from queue and marked as cancelled.")
                if self._current_processing_record_id == execution_record_id:
                    self._current_processing_record_id = None
                    self._try_process_next_task_from_queue()
                return True
        logger.warning(f"Record ID {execution_record_id} not found for cancellation (neither processing nor queued).")
        return False

    def shutdown(self) -> None:
        logger.info("WorkflowEngine shutting down...")
        with self._queue_lock:
            logger.info(f"Cancelling {len(self._task_queue)} tasks from queue during shutdown.")
            while self._task_queue:
                queued_item_to_cancel = self._task_queue.popleft()
                record = queued_item_to_cancel.execution_record
                queued_item_to_cancel.cancel_event.set()
                record.fail("Task cancelled from queue due to engine shutdown")
                history_manager.save_record(record)
        active_tasks_to_cancel = list(self._running_tasks.items())
        if active_tasks_to_cancel: logger.info(f"Signaling cancellation for {len(active_tasks_to_cancel)} tasks in executor during shutdown.")
        for record_id_shut, (_future_shut, record_shut, cancel_event_shut) in active_tasks_to_cancel:
            if record_shut.status not in TERMINAL_STATUSES:
                logger.info(f"Signaling shutdown cancel for Record ID: {record_id_shut}")
                cancel_event_shut.set()
        logger.info("Shutting down ThreadPoolExecutor, waiting for tasks to complete or cancel...")
        self.executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down complete.")
        with self._queue_lock: self._current_processing_record_id = None

workflow_engine = WorkflowEngine()
