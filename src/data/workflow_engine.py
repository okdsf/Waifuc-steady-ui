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

from .workflow import Workflow, WorkflowStep
from .execution_history import ExecutionRecord, history_manager
from src.tools.actions.action_registry import registry as action_registry
from src.tools.sources.source_registry import registry as source_registry
from src.tools.actions.waifuc_actions import WaifucActionWrapper
from waifuc.source import LocalSource

# 新增：定义全局 logger
logger = logging.getLogger(__name__)

class WorkflowEngine:
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running_tasks = {}
        os.makedirs("logs", exist_ok=True)

    def execute_workflow(self, workflow: Workflow,
                       source_type: str, source_params: Dict[str, Any],
                       output_directory: str,
                       progress_callback: Callable[[str, float, str], None] = None) -> ExecutionRecord:
        record = history_manager.create_record(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            source_type=source_type,
            source_params=source_params,
            output_directory=output_directory
        )
        cancel_event = threading.Event()
        future = self.executor.submit(
            self._execute_workflow_internal,
            workflow, source_type, source_params, output_directory,
            record, progress_callback, cancel_event
        )
        self._running_tasks[record.id] = (future, record, cancel_event)
        return record

    def _execute_workflow_internal(self, workflow: Workflow,
                                  source_type: str, source_params: Dict[str, Any],
                                  output_directory: str, record: ExecutionRecord,
                                  progress_callback: Callable[[str, float, str], None] = None,
                                  cancel_event: threading.Event = None) -> None:
        class CancelledError(Exception):
            pass

        def clean_metadata(directory):
            try:
                for filename in os.listdir(directory):
                    if filename.startswith('.') and filename.endswith('_meta.json'):
                        os.remove(os.path.join(directory, filename))
            except Exception as e:
                logger.warning(f"清理元数据失败: {str(e)}")

        try:
            os.makedirs(output_directory, exist_ok=True)
            temp_dir = tempfile.mkdtemp()
            temp_input_dir = os.path.join(temp_dir, 'input')
            os.makedirs(temp_input_dir, exist_ok=True)

            try:
                log_file = os.path.join("logs", f"{record.id}_log.txt")
                file_handler = logging.FileHandler(log_file, 'w', 'utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                task_logger = logging.getLogger(f"workflow.{record.id}")
                task_logger.setLevel(logging.INFO)
                task_logger.addHandler(file_handler)

                if progress_callback:
                    progress_callback("获取图像", 0.0, "准备图像来源...")
                task_logger.info(f"开始执行工作流: {workflow.name}")
                task_logger.info(f"图像来源: {source_type}")
                task_logger.info(f"输出目录: {output_directory}")

                if cancel_event and cancel_event.is_set():
                    raise CancelledError("任务被取消")

                try:
                    source = source_registry.create_source(source_type, **source_params)
                    record.add_step_log("source", source_type, "started", "创建图像来源")
                    task_logger.info("从来源获取图像...")
                    if progress_callback:
                        progress_callback("获取图像", 0.1, "正在获取图像...")
                    if source_type == "LocalSource":
                        input_dir = source_params.get("directory", "")
                        if not os.path.exists(input_dir):
                            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
                        total_files = sum(1 for f in os.listdir(input_dir)
                                        if os.path.isfile(os.path.join(input_dir, f)) and
                                        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                        record.total_images = total_files
                        task_logger.info(f"发现 {total_files} 个图像文件")
                    else:
                        task_logger.info("开始下载图像...")
                        if progress_callback:
                            progress_callback("获取图像", 0.2, "下载图像...")
                        from waifuc.export import SaveExporter
                        source.source.export(SaveExporter(temp_input_dir))
                        total_files = sum(1 for f in os.listdir(temp_input_dir)
                                        if os.path.isfile(os.path.join(temp_input_dir, f)) and
                                        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')))
                        record.total_images = total_files
                        task_logger.info(f"已下载 {total_files} 个图像文件")
                        input_dir = temp_input_dir
                    record.add_step_log("source", source_type, "completed",
                                        f"成功获取 {record.total_images} 个图像文件")
                except Exception as e:
                    error_msg = f"获取图像失败: {str(e)}"
                    task_logger.error(error_msg)
                    record.add_step_log("source", source_type, "failed", error_msg)
                    record.fail(error_msg)
                    if progress_callback:
                        progress_callback("错误", 0, error_msg)
                    return

                if cancel_event and cancel_event.is_set():
                    raise CancelledError("任务被取消")

                current_dir = input_dir
                success_count = 0
                failed_count = 0

                for i, step in enumerate(workflow.steps):
                    step_progress_base = 0.3 + (i / len(workflow.steps)) * 0.6
                    unique_id = uuid.uuid4().hex[:8]
                    step_output_dir = os.path.join(temp_dir, f"step_{i+1}_{unique_id}")
                    os.makedirs(step_output_dir, exist_ok=True)
                    task_logger.info(f"执行步骤 {i+1}/{len(workflow.steps)}: {step.action_name}")
                    task_logger.info(f"步骤 {i+1} 输入目录: {current_dir}")
                    task_logger.info(f"步骤 {i+1} 输出目录: {step_output_dir}")
                    record.add_step_log(step.id, step.action_name, "started",
                                       f"开始执行步骤 {i+1}/{len(workflow.steps)}")
                    if progress_callback:
                        progress_callback("处理图像", step_progress_base,
                                         f"执行步骤 {i+1}/{len(workflow.steps)}: {step.action_name}")

                    if cancel_event and cancel_event.is_set():
                        raise CancelledError("任务被取消")

                    try:
                        action = action_registry.create_action(step.action_name, **step.params)
                        if isinstance(action, WaifucActionWrapper) and hasattr(action, 'action'):
                            action_instance = action.action
                        else:
                            action_instance = action

                        if False and step.action_name == "PreSortImagesAction": # 逻辑上永远为假

                            pass
                        elif False and step.action_name == "EnhancedImageProcessAction": # 逻辑上永远为假
                        
                            pass
                        else: 

                            source = LocalSource(current_dir)
                            processed = source.attach(action_instance)
                            from waifuc.export import SaveExporter
                            processed.export(SaveExporter(step_output_dir))
                            clean_metadata(step_output_dir)


                        output_files = [f for f in os.listdir(step_output_dir)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
                        if not output_files:
                            task_logger.warning(f"步骤 {step.action_name} 未生成任何图像")
                        record.add_step_log(step.id, step.action_name, "completed",
                                           f"步骤 {i+1}/{len(workflow.steps)} 成功完成，生成 {len(output_files)} 张图像")
                        if progress_callback:
                            progress_callback("处理图像", step_progress_base + 0.6/len(workflow.steps),
                                            f"步骤 {i+1}/{len(workflow.steps)} 完成")
                        current_dir = step_output_dir

                    except Exception as e:
                        error_msg = f"步骤 {i+1} ({step.action_name}) 执行失败: {str(e)}"
                        task_logger.error(error_msg)
                        record.add_step_log(step.id, step.action_name, "failed", error_msg)
                        failed_count += 1
                        if i == 0:
                            record.fail(error_msg)
                            if progress_callback:
                                progress_callback("错误", 0, error_msg)
                            return
                        current_dir = os.path.join(temp_dir, f"step_{i}")

                if cancel_event and cancel_event.is_set():
                    raise CancelledError("任务被取消")

                if workflow.steps:
                    output_files_count = 0
                    for root, dirs, files in os.walk(current_dir):
                        if cancel_event and cancel_event.is_set():
                            raise CancelledError("任务被取消")
                        for filename in files:
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                                base, ext = os.path.splitext(filename)
                                unique_filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                                src_path = os.path.join(root, filename)
                                dst_path = os.path.join(output_directory, unique_filename)
                                try:
                                    shutil.copy2(src_path, dst_path)
                                    output_files_count += 1
                                except Exception as e:
                                    task_logger.error(f"复制最终文件失败: {src_path} -> {dst_path}, 错误: {str(e)}")
                    task_logger.info(f"已将 {output_files_count} 个文件复制到 {output_directory}")
                    clean_metadata(output_directory)

                success_count = record.total_images - failed_count
                record.complete(
                    total_images=record.total_images,
                    processed_images=record.total_images,
                    success_images=success_count,
                    failed_images=failed_count
                )
                history_manager.save_record(record)
                task_logger.info(f"工作流执行完成. 总图像: {record.total_images}, "
                          f"成功: {success_count}, 失败: {failed_count}")
                if progress_callback:
                    progress_callback("完成", 1.0,
                                    f"处理完成. 总图像: {record.total_images}, "
                                    f"成功: {success_count}, 失败: {failed_count}")

            except CancelledError as e:
                error_msg = str(e)
                task_logger.info(error_msg)
                record.fail(error_msg)
                history_manager.save_record(record)
                if progress_callback:
                    progress_callback("取消", 0.0, error_msg)
                return

            finally:
                shutil.rmtree(temp_dir)
                file_handler.close()
                task_logger.removeHandler(file_handler)

        except Exception as e:
            error_msg = f"工作流执行出错: {str(e)}"
            task_logger.error(error_msg)
            record.fail(error_msg)
            history_manager.save_record(record)
            if progress_callback:
                progress_callback("错误", 0, error_msg)

        finally:
            if record.id in self._running_tasks:
                del self._running_tasks[record.id]

    def get_running_tasks(self) -> Dict[str, ExecutionRecord]:
        running_records = {}
        for task_id, (future, record, cancel_event) in list(self._running_tasks.items()):
            if future.done():
                del self._running_tasks[task_id]
            else:
                running_records[task_id] = record
        return running_records

    def cancel_task(self, task_id: str) -> bool:
        if task_id not in self._running_tasks:
            logger.warning(f"Task {task_id} not found in running tasks")
            return False
        future, record, cancel_event = self._running_tasks[task_id]
        cancelled = future.cancel()
        cancel_event.set()
        if cancelled:
            record.fail("任务被取消")
            history_manager.save_record(record)
            del self._running_tasks[task_id]
            logger.info(f"Task {task_id} cancelled via future.cancel")
        else:
            logger.info(f"Task {task_id} marked for cancellation via cancel_event")
        return True

    def shutdown(self) -> None:
        for task_id, (future, record, cancel_event) in list(self._running_tasks.items()):
            cancel_event.set()
            future.cancel()
            record.fail("任务被取消（引擎关闭）")
            history_manager.save_record(record)
        self.executor.shutdown(wait=True)

workflow_engine = WorkflowEngine()

