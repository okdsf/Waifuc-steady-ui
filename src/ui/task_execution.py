import os
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTextEdit, QTreeWidget, QTreeWidgetItem, QMessageBox, QGroupBox,
    QSplitter, QFrame, QToolBar, QAction, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from src.data import Workflow, ExecutionRecord, workflow_engine, history_manager


class TaskExecutionWidget(QWidget):
    """
    任务执行部件，展示处理任务的执行过程和结果
    """
    task_started = pyqtSignal()
    task_finished = pyqtSignal(bool, str)  # 成功状态, 消息

    # 新增信号，用于线程安全更新UI
    progress_signal = pyqtSignal(str, float, str)  # status, progress, message
    log_signal = pyqtSignal(str)  # message
    result_signal = pyqtSignal(dict)  # step_log

    def __init__(self, workflow: Workflow, source_type: str, source_params: Dict[str, Any],
                output_directory: str, parent=None):
        super().__init__(parent)

        self.workflow = workflow
        self.source_type = source_type
        self.source_params = source_params
        self.output_directory = output_directory

        self.execution_record: Optional[ExecutionRecord] = None

        # 初始化UI
        self.init_ui()

        # 连接信号到槽
        self.progress_signal.connect(self.on_progress_update)
        self.log_signal.connect(self.add_log)
        # self.result_signal.connect(self.update_result_tree) # 移除执行结果信号连接

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        return bool(self.execution_record and 
                   self.execution_record.status in ["running", "processing"])

    def can_stop(self) -> bool:
        """检查任务是否可以停止"""
        return bool(self.execution_record and 
                   self.execution_record.status in ["running", "processing"])

    def is_queued(self) -> bool:
        """检查任务是否在队列中"""
        return bool(self.execution_record and 
                   self.execution_record.status == "queued")

    def is_completed(self) -> bool:
        """检查任务是否已完成"""
        return bool(self.execution_record and 
                   self.execution_record.status in ["completed", "failed", "cancelled"])

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 任务信息
        info_layout = QHBoxLayout()

        self.workflow_label = QLabel(self.tr("工作流: ") + self.workflow.name)
        info_layout.addWidget(self.workflow_label)

        self.source_label = QLabel(self.tr("源: ") + self.source_type)
        info_layout.addWidget(self.source_label)

        self.output_label = QLabel(self.tr("输出: ") + self.output_directory)
        info_layout.addWidget(self.output_label)

        # 添加到主布局
        layout.addLayout(info_layout)

        # 进度部分
        progress_group = QGroupBox(self.tr("执行进度"))
        progress_layout = QVBoxLayout(progress_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel(self.tr("准备执行..."))
        progress_layout.addWidget(self.status_label)

        # 添加到主布局
        layout.addWidget(progress_group)

        # 日志输出
        log_group = QGroupBox(self.tr("执行日志"))
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        # 添加到主布局
        layout.addWidget(log_group)

        # 结果部分
        # result_group = QGroupBox(self.tr("执行结果"))
        # result_layout = QVBoxLayout(result_group)

        # self.result_tree = QTreeWidget()
        # self.result_tree.setHeaderLabels([self.tr("步骤"), self.tr("状态"), self.tr("详情")])
        # self.result_tree.setColumnWidth(0, 200)
        # self.result_tree.setColumnWidth(1, 100)
        # result_layout.addWidget(self.result_tree)

        # 添加到主布局
        # layout.addWidget(result_group)

        # 控制按钮
        buttons_layout = QHBoxLayout()

        self.start_button = QPushButton(self.tr("开始执行"))
        self.start_button.clicked.connect(self.start_task)
        buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton(self.tr("停止执行"))
        self.stop_button.clicked.connect(self.stop_task)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)

        self.open_output_button = QPushButton(self.tr("打开输出目录"))
        self.open_output_button.clicked.connect(self.open_output_directory)
        buttons_layout.addWidget(self.open_output_button)

        # 添加到主布局
        layout.addLayout(buttons_layout)

        # 更新计时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_progress)
        self.update_timer.setInterval(1000)  # 1秒更新一次

    def retranslateUi(self):
        self.workflow_label.setText(self.tr("工作流: ") + self.workflow.name)
        self.source_label.setText(self.tr("源: ") + self.source_type)
        self.output_label.setText(self.tr("输出: ") + self.output_directory)
        # 分组框
        self.findChild(QGroupBox, None).setTitle(self.tr("执行进度")) if self.findChild(QGroupBox, None) else None
        # 进度条状态
        self.status_label.setText(self.tr("准备执行..."))
        # 日志分组
        # 由于QGroupBox没有objectName，需按顺序设置
        group_boxes = self.findChildren(QGroupBox)
        if len(group_boxes) > 1:
            group_boxes[1].setTitle(self.tr("执行日志"))
        # 移除执行结果栏相关
        # if len(group_boxes) > 2:
        #     group_boxes[2].setTitle(self.tr("执行结果"))
        # 结果树表头
        # self.result_tree.setHeaderLabels([
        #     self.tr("步骤"), self.tr("状态"), self.tr("详情")
        # ])
        # 按钮
        self.start_button.setText(self.tr("开始执行"))
        self.stop_button.setText(self.tr("停止执行"))
        self.open_output_button.setText(self.tr("打开输出目录"))

    def start_task(self):
        """开始执行任务"""
        if self.is_running():
            return

        # 重置UI
        self.progress_bar.setValue(0)
        self.status_label.setText(self.tr("正在启动..."))
        self.log_text.clear()
        # self.result_tree.clear() # 移除结果树清空

        # 更新按钮状态
        self.start_button.setEnabled(False)
        # self.stop_button.setEnabled(False) # 移除停止按钮状态更新

        # 添加首条日志
        self.log_signal.emit(self.tr("开始执行任务..."))
        self.log_signal.emit(self.tr("工作流: ") + self.workflow.name)
        self.log_signal.emit(self.tr("源类型: ") + self.source_type)
        self.log_signal.emit(self.tr("输出目录: ") + self.output_directory)

        # 启动任务
        self.execution_record = workflow_engine.execute_workflow(
            self.workflow,
            self.source_type,
            self.source_params,
            self.output_directory,
            self.on_progress_callback
        )

        # 注册状态变更回调
        if self.execution_record:
            self.execution_record.subscribe_status(self.on_status_changed)

        # 启动更新计时器
        self.update_timer.start()

        # 发送任务开始信号
        self.task_started.emit()

        # 主动刷新主窗口的停止按钮状态
        main_window = self.parent()
        from PyQt5.QtWidgets import QMainWindow
        while main_window and not isinstance(main_window, QMainWindow):
            main_window = main_window.parent()
        if main_window and hasattr(main_window, 'update_stop_button_state'):
            main_window.update_stop_button_state()

    def on_status_changed(self, status):
        from PyQt5.QtWidgets import QMainWindow
        main_window = self.parent()
        while main_window and not isinstance(main_window, QMainWindow):
            main_window = main_window.parent()
        if main_window and hasattr(main_window, 'update_stop_button_state'):
            main_window.update_stop_button_state()

    def stop_task(self):
        if not self.can_stop():
            QMessageBox.information(self, self.tr("提示"), self.tr("当前任务没有在运行。"))
            return

        reply = QMessageBox.question(
            self, self.tr("确认停止"), self.tr("确定要停止当前任务吗？"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            if workflow_engine.cancel_task(self.execution_record.id):
                self.log_signal.emit(self.tr("正在取消任务..."))
                self.status_label.setText(self.tr("正在取消..."))
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.update_timer.stop()
                self.task_finished.emit(False, self.tr("任务已取消"))
            else:
                record = history_manager.get_record(self.execution_record.id)
                if record and record.status in ["completed", "failed", "cancelled"]:
                    self.log_signal.emit(self.tr("任务已经完成或已取消"))
                    QMessageBox.information(self, self.tr("提示"), self.tr("任务已经完成或已取消。"))
                else:
                    self.log_signal.emit(self.tr("无法取消任务"))
                    QMessageBox.warning(self, self.tr("错误"), self.tr("无法取消任务，请稍后重试。"))
        except Exception as e:
            self.log_signal.emit(self.tr(f"停止任务时发生错误: {str(e)}"))
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"停止任务时发生错误: {str(e)}"))

    def on_progress_callback(self, status: str, progress: float, message: str):
        """
        进度更新回调，子线程调用，发信号给主线程

        Args:
            status: 当前状态
            progress: 进度值 (0.0 - 1.0)
            message: 状态消息
        """
        self.progress_signal.emit(status, progress, message)

    @pyqtSlot(str, float, str)
    def on_progress_update(self, status: str, progress: float, message: str):
        """
        进度更新槽函数，主线程更新UI

        Args:
            status: 当前状态
            progress: 进度值 (0.0 - 1.0)
            message: 状态消息
        """
        # 更新进度条
        progress_value = int(progress * 100)
        self.progress_bar.setValue(progress_value)

        # 更新状态标签
        self.status_label.setText(self.tr(f"{status}: {message}"))

        # 添加日志
        self.log_signal.emit(self.tr(f"[{status}] {message}"))

    @pyqtSlot(str)
    def add_log(self, message: str):
        """
        添加日志，主线程更新UI

        Args:
            message: 日志消息
        """
        self.log_text.append(message)

        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    # 移除result_tree相关方法和信号
    # @pyqtSlot(dict)
    # def update_result_tree(self, step_log: dict):
    #     """
    #     更新结果树显示，主线程更新UI
    #
    #     Args:
    #         step_log: 步骤日志字典
    #     """
    #     step_id = step_log.get("step_id")
    #     step_name = step_log.get("step_name")
    #     status = step_log.get("status")
    #     message = step_log.get("message", "")

    #     # 创建或获取步骤项
    #     found = False
    #     for i in range(self.result_tree.topLevelItemCount()):
    #         item = self.result_tree.topLevelItem(i)
    #         if item.text(0) == step_name:
    #             # 更新已有项
    #             item.setText(1, status)
    #             item.setText(2, message)
    #             step_item = item
    #             found = True
    #             break

    #     if not found:
    #         # 创建新项
    #         step_item = QTreeWidgetItem([step_name, status, message])
    #         self.result_tree.addTopLevelItem(step_item)

    #     # 添加详情子项
    #     details = step_log.get("details", {})
    #     if details:
    #         for key, value in details.items():
    #             detail_item = QTreeWidgetItem([key, "", str(value)])
    #             step_item.addChild(detail_item)

    #     # 展开所有项
    #     self.result_tree.expandAll()

    def check_task_status(self):
        """检查任务状态并更新UI"""
        if not self.execution_record:
            return
            
        try:
            record = history_manager.get_record(self.execution_record.id)
            if not record:
                return
                
            # 更新执行记录
            self.execution_record = record
            
            # 检查任务状态是否发生变化
            if record.status != "running" and self.is_running():
                # 任务已完成或失败
                self.update_timer.stop()
                self.start_button.setEnabled(True)
                # self.stop_button.setEnabled(False) # 移除停止按钮状态更新
                
                # 根据状态更新UI
                if record.status == "completed":
                    self.progress_bar.setValue(100)
                    self.status_label.setText(self.tr(f"已完成 (成功: {record.success_images}, 失败: {record.failed_images})"))
                    self.log_signal.emit(self.tr(f"任务完成. 总图像: {record.total_images}, 成功: {record.success_images}, 失败: {record.failed_images}"))
                    self.task_finished.emit(True, self.tr(f"成功处理 {record.success_images} 个图像, 失败 {record.failed_images} 个"))
                elif record.status == "failed":
                    if record.error_message and ("cancelled" in record.error_message.lower() or "取消" in record.error_message):
                        self.status_label.setText(self.tr("已取消"))
                        self.log_signal.emit(self.tr("任务已取消"))
                        self.task_finished.emit(False, self.tr("任务已取消"))
                    else:
                        self.status_label.setText(self.tr(f"出错: {record.error_message}"))
                        self.log_signal.emit(self.tr(f"任务失败: {record.error_message}"))
                        self.task_finished.emit(False, record.error_message or self.tr("任务执行失败"))
                elif record.status == "cancelled":
                    self.status_label.setText(self.tr("已取消"))
                    self.log_signal.emit(self.tr("任务已取消"))
                    self.task_finished.emit(False, self.tr("任务已取消"))
                else:
                    self.status_label.setText(self.tr(f"状态: {record.status}"))
                    self.log_signal.emit(self.tr(f"任务状态: {record.status}"))
                    self.task_finished.emit(False, self.tr(f"任务状态: {record.status}"))
        except Exception as e:
            self.log_signal.emit(self.tr(f"检查任务状态时发生错误: {str(e)}"))

    def update_progress(self):
        """更新进度和结果显示"""
        if not self.execution_record:
            return

        # 直接使用自己的execution_record，不依赖history_manager
        record = self.execution_record

        # 根据执行记录状态更新UI
        if record.status == "running":
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        elif record.status in ["completed", "failed", "cancelled"]:
            self.update_timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            if record.status == "completed":
                self.progress_bar.setValue(100)
                self.status_label.setText(self.tr(f"已完成 (成功: {record.success_images}, 失败: {record.failed_images})"))
                self.log_signal.emit(self.tr(f"任务完成. 总图像: {record.total_images}, 成功: {record.success_images}, 失败: {record.failed_images}"))
                self.task_finished.emit(True, self.tr(f"成功处理 {record.success_images} 个图像, 失败 {record.failed_images} 个"))
            elif record.status == "failed":
                if record.error_message and ("cancelled" in record.error_message.lower() or "取消" in record.error_message):
                    self.status_label.setText(self.tr("已取消"))
                    self.log_signal.emit(self.tr("任务已取消"))
                    self.task_finished.emit(False, self.tr("任务已取消"))
                else:
                    self.status_label.setText(self.tr(f"出错: {record.error_message}"))
                    self.log_signal.emit(self.tr(f"任务失败: {record.error_message}"))
                    self.task_finished.emit(False, record.error_message or self.tr("任务执行失败"))
            else:
                self.status_label.setText(self.tr(f"状态: {record.status}"))
                self.log_signal.emit(self.tr(f"任务状态: {record.status}"))
                self.task_finished.emit(False, self.tr(f"任务状态: {record.status}"))
        elif record.status == "queued":
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.status_label.setText(self.tr("等待执行..."))

    def open_output_directory(self):
        """打开输出目录"""
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                os.startfile(self.output_directory)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", self.output_directory])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.output_directory])
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("错误"), self.tr(f"无法打开输出目录: {str(e)}")
            )