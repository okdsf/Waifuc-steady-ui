import os
import logging
import tempfile
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from PIL import Image # 假设您会用到，根据您原始文件
import threading
import collections

from .workflow import Workflow, WorkflowStep
from .execution_history import ExecutionRecord, history_manager
from src.tools.actions.action_registry import registry as action_registry
from src.tools.sources.source_registry import registry as source_registry
from src.tools.actions.waifuc_actions import WaifucActionWrapper
from waifuc.source import LocalSource

logger = logging.getLogger(__name__)

# 定义终结状态集合，请根据您的 ExecutionRecord.status 的实际值进行调整
# 这些是任务达到最终状态后，不应再被修改的状态。
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
# 如果您的 record.fail("任务被取消") 会将 status 设为 "cancelled"，请确保 "cancelled" 在这里。
# 如果 record.fail() 总是将 status 设为 "failed"，那么 "cancelled" 可能不需要，
# 或者您需要一个更具体的 "cancelled_from_queue" 等状态。
# 为了安全，我包含了 "completed", "failed", "cancelled"。

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
        record.status = "queued" # 初始状态
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

    def _on_task_done(self, future_object, record_id: str): # Line ~160
        try:
            exception = future_object.exception()
            if exception:
                logger.error(f"Task (Record ID: {record_id}) execution raised an exception in future: {exception}", exc_info=exception)
                if record_id in self._running_tasks:
                    _, record, _ = self._running_tasks[record_id]
                    # --- MODIFICATION HERE ---
                    if record.status not in TERMINAL_STATUSES: # Line ~164 where error occurred
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

            # --- 您原始的 _execute_workflow_internal 核心代码应该在这里 ---
            # 以下是根据您之前文件和我的理解重建的，请务必核对
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
                    from waifuc.export import SaveExporter
                    source.source.export(SaveExporter(temp_input_dir))
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

            for i, step in enumerate(workflow.steps):
                step_progress_base = 0.3 + (i / len(workflow.steps)) * 0.6
                unique_id_for_step_dir = uuid.uuid4().hex[:8]
                step_output_dir = os.path.join(temp_dir, f"step_{i+1}_{unique_id_for_step_dir}")
                os.makedirs(step_output_dir, exist_ok=True)
                task_logger.info(f"执行步骤 {i+1}/{len(workflow.steps)}: {step.action_name} (In: {current_dir_for_steps}, Out: {step_output_dir})")
                record.add_step_log(step.id, step.action_name, "started", f"开始执行步骤 {i+1}/{len(workflow.steps)}")
                if progress_callback: progress_callback("处理图像", step_progress_base, f"执行步骤 {i+1}/{len(workflow.steps)}: {step.action_name}")
                if cancel_event and cancel_event.is_set(): raise CancelledError(f"任务在步骤 {step.action_name} 执行前被取消")
                try:
                    action = action_registry.create_action(step.action_name, **step.params)
                    action_instance = action.action if isinstance(action, WaifucActionWrapper) and hasattr(action, 'action') else action
                    source_for_step = LocalSource(current_dir_for_steps)
                    processed_output = source_for_step.attach(action_instance)
                    from waifuc.export import SaveExporter
                    processed_output.export(SaveExporter(step_output_dir))
                    def clean_metadata_local(directory_to_clean):
                        try:
                            for filename in os.listdir(directory_to_clean):
                                if filename.startswith('.') and filename.endswith('_meta.json'):
                                    os.remove(os.path.join(directory_to_clean, filename))
                        except Exception as e_clean: logger.warning(f"清理元数据失败 ({directory_to_clean}): {str(e_clean)}")
                    clean_metadata_local(step_output_dir)
                    output_files_count_step = len([f for f in os.listdir(step_output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))])
                    if not output_files_count_step: task_logger.warning(f"步骤 {step.action_name} 未生成任何图像")
                    record.add_step_log(step.id, step.action_name, "completed", f"步骤 {i+1}/{len(workflow.steps)} 成功完成，生成 {output_files_count_step} 张图像")
                    if progress_callback: progress_callback("处理图像", step_progress_base + 0.6/len(workflow.steps), f"步骤 {i+1}/{len(workflow.steps)} 完成")
                    current_dir_for_steps = step_output_dir
                except Exception as e_step:
                    error_msg_step = f"步骤 {i+1} ({step.action_name}) 执行失败: {str(e_step)}"
                    task_logger.error(error_msg_step, exc_info=True)
                    record.add_step_log(step.id, step.action_name, "failed", error_msg_step)
                    record.fail(f"工作流因步骤 {step.action_name} 失败而中止: {error_msg_step}") # 步骤失败导致工作流失败
                    history_manager.save_record(record)
                    if progress_callback: progress_callback("错误", step_progress_base, f"步骤 {step.action_name} 失败，工作流中止")
                    return # 终止工作流
            
            if cancel_event and cancel_event.is_set(): raise CancelledError("任务在复制最终文件前被取消")
            final_output_files_count = 0
            if workflow.steps:
                for root, _dirs, files_in_final_step_dir in os.walk(current_dir_for_steps):
                    if cancel_event and cancel_event.is_set(): raise CancelledError("复制操作被取消")
                    for filename_final in files_in_final_step_dir:
                        if filename_final.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            base, ext = os.path.splitext(filename_final)
                            unique_filename_final = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                            src_path_final = os.path.join(root, filename_final)
                            dst_path_final = os.path.join(output_directory, unique_filename_final)
                            try:
                                shutil.copy2(src_path_final, dst_path_final)
                                final_output_files_count += 1
                            except Exception as e_copy: task_logger.error(f"复制最终文件失败: {src_path_final} -> {dst_path_final}, 错误: {str(e_copy)}")
                task_logger.info(f"已将 {final_output_files_count} 个文件复制到最终输出目录: {output_directory}")
                def clean_metadata_local_final(directory_to_clean):
                    try:
                        for filename_clean in os.listdir(directory_to_clean):
                            if filename_clean.startswith('.') and filename_clean.endswith('_meta.json'):
                                os.remove(os.path.join(directory_to_clean, filename_clean))
                    except Exception as e_clean_final: logger.warning(f"清理最终目录元数据失败 ({directory_to_clean}): {str(e_clean_final)}")
                clean_metadata_local_final(output_directory)

            record.complete(
                total_images=record.total_images if record.total_images is not None else 0,
                processed_images=final_output_files_count,
                success_images=final_output_files_count,
                failed_images=0 # Assuming if we reach here, there were no earlier critical failures
            )
            history_manager.save_record(record)
            # --- MODIFICATION HERE ---
            # Line ~431 where error occurred
            completion_log_message = getattr(record, 'message', record.status) # Use message if available, else status
            task_logger.info(f"工作流执行完成. Record ID: {record.id}. 状态: {record.status}, 详情: {completion_log_message}")
            if progress_callback: progress_callback("完成", 1.0, f"处理完成. 总图像: {record.total_images}, 成功: {final_output_files_count}")

        except CancelledError as e:
            error_msg = str(e)
            task_logger.info(f"Record ID {record.id}: {error_msg} (Caught in _execute_workflow_internal)")
            # 确保 fail 方法正确设置 status，例如为 "cancelled" 或 "failed"
            record.fail(error_msg) # fail 方法应该更新 record.status
            history_manager.save_record(record)
            if progress_callback:
                progress_val = 0.0
                progress_callback("取消", progress_val, error_msg)
        
        except Exception as e: # Line ~450
            error_msg = f"工作流执行中发生意外错误: {str(e)}"
            task_logger.error(f"Record ID {record.id}: {error_msg}", exc_info=True)
            # --- MODIFICATION HERE ---
            if record.status not in TERMINAL_STATUSES: # Line ~452 where error occurred
                record.fail(error_msg)
                history_manager.save_record(record)
            if progress_callback:
                progress_callback("错误", 0.0, error_msg)
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
            # --- MODIFICATION HERE ---
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
                # fail() should set status to 'failed' or 'cancelled'
                record_to_cancel.fail("任务已从队列中取消") # This should set status to one of TERMINAL_STATUSES
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
                record.fail("任务因引擎关闭而从队列取消") # This should set status to one of TERMINAL_STATUSES
                history_manager.save_record(record)
        active_tasks_to_cancel = list(self._running_tasks.items())
        if active_tasks_to_cancel: logger.info(f"Signaling cancellation for {len(active_tasks_to_cancel)} tasks in executor during shutdown.")
        for record_id_shut, (_future_shut, record_shut, cancel_event_shut) in active_tasks_to_cancel:
            # --- MODIFICATION HERE ---
            if record_shut.status not in TERMINAL_STATUSES:
                logger.info(f"Signaling shutdown cancel for Record ID: {record_id_shut}")
                cancel_event_shut.set()
        logger.info("Shutting down ThreadPoolExecutor, waiting for tasks to complete or cancel...")
        self.executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down complete.")
        with self._queue_lock: self._current_processing_record_id = None


workflow_engine = WorkflowEngine()