"""
工作流设计器模块 - 可视化设计图像处理工作流
"""
import os
import json
from typing import Optional, List, Dict, Any, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QTreeWidget, QTreeWidgetItem, QMessageBox, QDialog,
    QLineEdit, QTextEdit, QFormLayout, QDialogButtonBox, QFileDialog,
    QGroupBox, QSplitter, QFrame, QToolBar, QAction, QMenu, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QMimeData
from PyQt5.QtGui import QIcon, QDrag, QPixmap

from src.data import Workflow, WorkflowStep, workflow_manager
from src.tools.actions.action_registry import registry as action_registry


class StepConfigDialog(QDialog):
    """
    操作步骤配置对话框
    """
    def __init__(self, action_name: str, params: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self.action_name = action_name
        self.params = params or {}
        self.param_widgets = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"配置 {self.action_name}")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)

        # 添加表单布局
        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        # 创建参数输入控件
        self.create_param_widgets()

        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)




    def create_param_widgets(self):
        """创建参数输入控件"""
        action_default_params = action_registry.get_action_params(self.action_name)


        params_to_display = {**action_default_params, **self.params}

   
        for param_name, default_config_value in action_default_params.items():

            current_value_to_display = params_to_display.get(param_name)

            label = QLabel(f"{param_name}:")

 

            if default_config_value is None:
                widget = QLineEdit()
                if current_value_to_display is not None: 
                    widget.setText(str(current_value_to_display))
                else: # 否则，如果是必填项，显示占位符
                    widget.setPlaceholderText(f"请输入 {param_name}（必填）")
            
            elif isinstance(default_config_value, bool):
                widget = QCheckBox()
                # 使用 current_value_to_display (转换为布尔型以确保类型正确)
                widget.setChecked(bool(current_value_to_display))
            
            elif isinstance(default_config_value, int):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000) # 范围可以根据需要调整
                # 使用 current_value_to_display (转换为整型)
                widget.setValue(int(current_value_to_display))
            
            elif isinstance(default_config_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1000000, 1000000) # 范围可以根据需要调整
                widget.setDecimals(4) # 小数位数可以根据需要调整
                # 使用 current_value_to_display (转换为浮点型)
                widget.setValue(float(current_value_to_display))
            
            elif isinstance(default_config_value, dict):
                widget = QLineEdit()
                # 显示 current_value_to_display 的字符串形式
                widget.setText(str(current_value_to_display))
                widget.setReadOnly(True)
                edit_button = QPushButton("编辑...")
                # edit_dict_param 方法会从 widget 读取文本，所以 setText(str(current_value_to_display)) 很重要
                edit_button.clicked.connect(lambda checked, name=param_name: self.edit_dict_param(name))
                
                self.form_layout.addRow(label, widget)
                self.form_layout.addRow("", edit_button)
                self.param_widgets[param_name] = (widget, edit_button)
                continue # 跳过此循环末尾的默认 addRow
            
            elif isinstance(default_config_value, (list, tuple)):
                widget = QLineEdit()
                # 显示 current_value_to_display 的字符串形式
                widget.setText(str(current_value_to_display))
                widget.setReadOnly(True)
                edit_button = QPushButton("编辑...")
                # edit_list_param 方法会从 widget 读取文本
                edit_button.clicked.connect(lambda checked, name=param_name: self.edit_list_param(name))

                self.form_layout.addRow(label, widget)
                self.form_layout.addRow("", edit_button)
                self.param_widgets[param_name] = (widget, edit_button)
                continue # 跳过此循环末尾的默认 addRow
            
            else: # 其他类型（如字符串）默认使用 QLineEdit
                widget = QLineEdit(str(current_value_to_display))

            self.form_layout.addRow(label, widget)
            self.param_widgets[param_name] = widget



    def edit_dict_param(self, param_name):
        """编辑字典参数"""
        widget = self.param_widgets[param_name][0]
        current_value = eval(widget.text())
        text = json.dumps(current_value, indent=2)
        new_text, ok = QInputDialog.getMultiLineText(self, f"编辑 {param_name}", "输入 JSON 格式数据:", text)
        if ok:
            try:
                new_value = json.loads(new_text)
                widget.setText(str(new_value))
            except json.JSONDecodeError:
                QMessageBox.warning(self, "格式错误", "输入的 JSON 格式不正确。")

    def edit_list_param(self, param_name):
        """编辑列表参数"""
        widget = self.param_widgets[param_name][0]
        current_text = widget.text()
        try:
            current_value = eval(current_text)
            if not isinstance(current_value, (list, tuple)):
                current_value = []
        except:
            current_value = []
        text = json.dumps(current_value)
        new_text, ok = QInputDialog.getMultiLineText(self, f"编辑 {param_name}", "输入 JSON 格式数据:", text)
        if ok:
            try:
                new_value = json.loads(new_text)
                widget.setText(str(new_value))
            except json.JSONDecodeError:
                QMessageBox.warning(self, "格式错误", "输入的 JSON 格式不正确。")

    def get_params(self) -> Dict[str, Any]:
        """获取用户输入的参数"""
        result = {}
        
        # 已知有默认值的参数列表
        known_optional_params = [
            "person_conf", "halfbody_conf", "head_conf", "eye_conf",
            "min_size", "prefix", "sizes", "color", "scale", "model",
            "conf_threshold", "iou_threshold", "level", "version", "ratings",
            "keep_original", "head_scale", "split_eyes", "eye_scale",
            "split_person", "keep_origin_tags", "ratios"
        ]
        
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, tuple):
                widget = widget[0]
                try:
                    value = eval(widget.text())
                    result[param_name] = value
                except:
                    pass
            elif isinstance(widget, QCheckBox):
                result[param_name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                result[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                result[param_name] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                # 如果参数为空但在已知可选参数列表中，则跳过验证
                if not text and widget.placeholderText().startswith("请输入"):
                    if param_name in known_optional_params:
                        continue  # 跳过这个参数，不添加到结果中
                    else:
                        raise ValueError(f"参数 {param_name} 为必填项")
                try:
                    if text and '.' in text:
                        result[param_name] = float(text)
                    elif text and text.isdigit():
                        result[param_name] = int(text)
                    else:
                        result[param_name] = text
                except ValueError:
                    result[param_name] = text
            else:
                result[param_name] = None
        return result


class WorkflowNameDialog(QDialog):
    """
    工作流名称和描述对话框
    """
    def __init__(self, name: str = "", description: str = "", parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("工作流信息")
        self.setMinimumSize(400, 250)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 名称和描述输入
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit(name)
        form_layout.addRow("名称:", self.name_edit)
        
        self.description_edit = QTextEdit()
        self.description_edit.setText(description)
        self.description_edit.setPlaceholderText("输入工作流描述...")
        form_layout.addRow("描述:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_name(self) -> str:
        """获取名称"""
        return self.name_edit.text().strip()
    
    def get_description(self) -> str:
        """获取描述"""
        return self.description_edit.toPlainText().strip()


class ActionsListWidget(QListWidget):
    """
    操作列表部件，显示所有可用的操作
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        
        # 加载所有操作
        self.load_actions()
    
    def load_actions(self):
        """加载所有可用的操作"""
        self.clear()
        
        # 获取所有操作类别和操作
        categories = action_registry.get_categories()
        
        # 按类别添加操作
        for category in categories:
            # 添加类别项
            category_item = QListWidgetItem(category)
            category_item.setFlags(Qt.ItemIsEnabled)
            category_item.setBackground(Qt.lightGray)
            self.addItem(category_item)
            
            # 添加该类别下的操作
            actions = action_registry.get_actions_in_category(category)
            for action_name in actions:
                item = QListWidgetItem(action_name)
                item.setData(Qt.UserRole, {
                    "type": "action",
                    "name": action_name,
                    "category": category
                })
                self.addItem(item)
    
    def startDrag(self, supportedActions):
        """处理拖动开始事件"""
        item = self.currentItem()
        if item is None:
            return
        
        # 获取项数据
        data = item.data(Qt.UserRole)
        if data is None or data.get("type") != "action":
            return
        
        # 创建MIME数据
        mime_data = QMimeData()
        mime_data.setText(json.dumps(data))
        
        # 创建拖动对象
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # 设置拖动时的图标（可选）
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        drag.setPixmap(pixmap)
        
        # 开始拖动
        drag.exec_(Qt.CopyAction)


class WorkflowStepsWidget(QTreeWidget):
    """
    工作流步骤部件，显示工作流中的步骤
    """
    step_edited = pyqtSignal(str, dict)  # 步骤ID, 更新后的参数
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置列
        self.setHeaderLabels(["步骤", "参数"])
        self.setColumnWidth(0, 200)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.InternalMove)
        
        # 连接信号
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def load_workflow(self, workflow: Workflow):
        """加载工作流步骤"""
        self.clear()
        
        for step in workflow.steps:
            self.add_step_item(step)
    


    def add_step_item(self, step: WorkflowStep) -> QTreeWidgetItem:
        """添加步骤项"""
        item = QTreeWidgetItem([step.action_name, str(step.params)])
        item.setData(0, Qt.UserRole, {
            "type": "step",
            "id": step.id,
            "action_name": step.action_name,
            "params": step.params
        })
        
        step_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled
        item.setFlags(step_flags)
        
        item.takeChildren() 
        for param_name, param_value in step.params.items():
            param_item = QTreeWidgetItem([param_name, str(param_value)])
            param_item.setFlags(Qt.ItemIsEnabled) 
            item.addChild(param_item)
        
        self.addTopLevelItem(item) 
        return item
    
    def dragEnterEvent(self, event):
        """处理拖动进入事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)
    
    def dragMoveEvent(self, event):
        """处理拖动移动事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)
    
    def dropEvent(self, event):
        """处理放置事件"""
        if event.mimeData().hasText():
            # 当前处理从 ActionsListWidget 拖入新操作的逻辑
            # 这部分通过 "add" 信号更新后端，看起来是正常的
            try:
                data = json.loads(event.mimeData().text())
                if data.get("type") == "action":
                    action_name = data.get("name")
                    
                    # 弹出配置对话框
                    dialog = StepConfigDialog(action_name, parent=self)
                    if dialog.exec_():
                        params = dialog.get_params()
                        step = WorkflowStep(action_name, params) # 创建步骤
                        
                        # 确定插入位置 (基于视觉放置)
                        target_item = self.itemAt(event.pos())
                        index_at_drop = self.indexFromItem(target_item) if target_item else self.indexFromItem(self.invisibleRootItem())
                        
                        # 发送添加步骤信号 (由 WorkflowDesignerWidget 处理)
                        self.step_edited.emit("add", {
                            "step": step.to_dict(),
                            "index": index_at_drop.row() if index_at_drop.isValid() and target_item else -1 # -1 表示追加
                        })
                        
                        # 注意：下面的 add_step_item 会将item添加到树的末尾。
                        # 后端会根据 index 插入，WorkflowDesignerWidget 可能需要刷新树或确保UI同步。
                        # 为了更精确的UI插入，可以修改 add_step_item 或在这里直接插入QTreeWidgetItem。
                        # 不过，当前问题核心是reorder。
                        item = self.add_step_item(step) # 在UI上添加
                        item.setExpanded(True)
                
                event.acceptProposedAction()
            except Exception: # 更具体的异常捕获可能更好
                event.ignore() # 如果处理失败，明确忽略事件
                # pass # 避免空 except

        elif event.source() == self and (event.possibleActions() & Qt.MoveAction):
            # 这是内部拖拽移动操作 (reordering existing steps)
            
            # 记录移动前的ID顺序，用于判断是否有实际顺序变化
            original_ids = []
            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                data = item.data(0, Qt.UserRole)
                if data and data.get("type") == "step":
                    original_ids.append(data.get("id"))

            # 让QTreeWidget的默认实现处理视觉上的项目移动
            super().dropEvent(event)

            if event.isAccepted():
                # 拖放被接受，视觉顺序已更新
                # 现在从UI获取新的步骤ID顺序
                new_ordered_ids = []
                for i in range(self.topLevelItemCount()):
                    item = self.topLevelItem(i)
                    data = item.data(0, Qt.UserRole) # QTreeWidgetItem的用户数据
                    if data and data.get("type") == "step":
                        new_ordered_ids.append(data.get("id"))
                
                # 只有当项目数量一致且顺序确实发生改变时，才发送信号
                if len(new_ordered_ids) == len(original_ids) and new_ordered_ids != original_ids:
                    self.step_edited.emit("reorder", {"ordered_ids": new_ordered_ids})
                # 如果顺序没变或者项目数不一致（不太可能在简单的内部移动中发生），则不执行任何操作
            # else: event was not accepted by superclass, do nothing.

        else:
            # 其他类型的拖放事件，也交给父类处理
            super().dropEvent(event)
    
    def on_item_double_clicked(self, item, column):
        """处理项双击事件"""
        # 获取项数据
        data = item.data(0, Qt.UserRole)
        if data is None or data.get("type") != "step":
            return
        
        # 获取步骤信息
        step_id = data.get("id")
        action_name = data.get("action_name")
        params = data.get("params", {})
        
        # 弹出配置对话框
        dialog = StepConfigDialog(action_name, params, parent=self)
        if dialog.exec_():
            new_params = dialog.get_params()
            
            # 更新项显示
            item.setText(1, str(new_params))
            
            # 更新子项
            item.takeChildren()  # 移除所有子项
            for param_name, param_value in new_params.items():
                param_item = QTreeWidgetItem([param_name, str(param_value)])
                item.addChild(param_item)
            
            # 发送步骤编辑信号
            self.step_edited.emit("update", {
                "id": step_id,
                "params": new_params
            })
    
    def show_context_menu(self, position):
        """显示上下文菜单"""
        item = self.itemAt(position)
        if item is None:
            return
        
        # 获取项数据
        data = item.data(0, Qt.UserRole)
        if data is None or data.get("type") != "step":
            return
        
        # 创建菜单
        menu = QMenu(self)
        
        # 编辑操作
        edit_action = menu.addAction("编辑...")
        edit_action.triggered.connect(lambda: self.on_item_double_clicked(item, 0))
        
        # 删除操作
        delete_action = menu.addAction("删除")
        delete_action.triggered.connect(lambda: self.delete_step(item))
        
        # 显示菜单
        menu.exec_(self.mapToGlobal(position))
    
    def delete_step(self, item):
        """删除步骤"""
        # 获取项数据
        data = item.data(0, Qt.UserRole)
        if data is None or data.get("type") != "step":
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除", "确定要删除此步骤吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 获取步骤ID
        step_id = data.get("id")
        
        # 发送删除步骤信号
        self.step_edited.emit("delete", {"id": step_id})
        
        # 从控件中移除
        index = self.indexOfTopLevelItem(item)
        self.takeTopLevelItem(index)


class WorkflowDesignerWidget(QWidget):
    """
    工作流设计器部件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 当前工作流
        self.current_workflow: Optional[Workflow] = None
        
        # 创建布局
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        
        # 工具栏
        toolbar = QToolBar()
        main_layout.addWidget(toolbar)
        
        # 新建工作流
        new_action = toolbar.addAction("新建")
        new_action.triggered.connect(self.create_new_workflow)
        
        # 编辑信息
        edit_info_action = toolbar.addAction("编辑信息")
        edit_info_action.triggered.connect(self.edit_workflow_info)
        
        toolbar.addSeparator()
        
        # 工作流信息
        self.info_label = QLabel("未打开工作流")
        toolbar.addWidget(self.info_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧 - 操作列表
        actions_group = QGroupBox("可用操作")
        actions_layout = QVBoxLayout(actions_group)
        
        self.actions_list = ActionsListWidget()
        actions_layout.addWidget(self.actions_list)
        
        splitter.addWidget(actions_group)
        
        # 右侧 - 工作流步骤
        steps_group = QGroupBox("工作流步骤")
        steps_layout = QVBoxLayout(steps_group)
        
        self.steps_tree = WorkflowStepsWidget()
        steps_layout.addWidget(self.steps_tree)
        
        # 添加控制按钮
        buttons_layout = QHBoxLayout()
        
        self.add_step_button = QPushButton("添加步骤")
        self.add_step_button.clicked.connect(self.add_step)
        buttons_layout.addWidget(self.add_step_button)
        
        self.remove_step_button = QPushButton("删除步骤")
        self.remove_step_button.clicked.connect(self.remove_step)
        buttons_layout.addWidget(self.remove_step_button)
        
        steps_layout.addLayout(buttons_layout)
        
        splitter.addWidget(steps_group)
        
        # 设置分割器初始大小
        splitter.setSizes([200, 400])
        
        # 连接信号
        self.steps_tree.step_edited.connect(self.on_step_edited)
    
    def create_new_workflow(self):
        """创建新工作流"""
        # 弹出对话框输入名称和描述
        dialog = WorkflowNameDialog(parent=self)
        if not dialog.exec_():
            return
        
        name = dialog.get_name()
        if not name:
            QMessageBox.warning(self, "错误", "工作流名称不能为空。")
            return
        
        description = dialog.get_description()
        
        # 创建工作流
        self.current_workflow = workflow_manager.create_workflow(name, description)
        
        # 更新UI
        self.update_workflow_info()
        self.steps_tree.clear()
    
    def open_workflow_dialog(self):
        """打开已保存的工作流"""
        # 获取所有工作流
        all_workflows = workflow_manager.get_all_workflows()
        if not all_workflows:
            QMessageBox.information(self, "提示", "没有保存的工作流。")
            return
        
        # 创建项目列表
        items = []
        for workflow in all_workflows:
            items.append(f"{workflow.name} ({workflow.id})")
        
        # 弹出选择对话框
        item, ok = QInputDialog.getItem(
            self, "打开工作流", "选择要打开的工作流:",
            items, 0, False
        )
        
        if not ok or not item:
            return
        
        # 获取工作流ID
        workflow_id = item.split("(")[-1].strip(")")
        
        # 加载工作流
        self.load_workflow(workflow_id)
    
    def load_workflow(self, workflow_id: str):
        """加载工作流"""
        workflow = workflow_manager.get_workflow(workflow_id)
        if workflow is None:
            QMessageBox.warning(self, "错误", f"未找到工作流 {workflow_id}。")
            return
        
        # 设置当前工作流
        self.current_workflow = workflow
        
        # 更新UI
        self.update_workflow_info()
        self.steps_tree.load_workflow(workflow)
    
    def save_workflow(self):
        """保存当前工作流"""
        if self.current_workflow is None:
            QMessageBox.warning(self, "错误", "没有可保存的工作流。")
            return
        
        # 保存工作流
        if workflow_manager.save_workflow(self.current_workflow):
            QMessageBox.information(
                self, "成功", f"工作流 '{self.current_workflow.name}' 已保存。"
            )
        else:
            QMessageBox.warning(
                self, "错误", f"保存工作流 '{self.current_workflow.name}' 失败。"
            )
    
    def export_workflow_dialog(self):
        """导出工作流到文件"""
        if self.current_workflow is None:
            QMessageBox.warning(self, "错误", "没有可导出的工作流。")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出工作流", f"{self.current_workflow.name}.json",
            "JSON 文件 (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            # 导出工作流
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_workflow.to_dict(), f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(
                self, "成功", f"工作流已导出到 {file_path}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "错误", f"导出工作流失败: {str(e)}"
            )
    
    def import_workflow_dialog(self):
        """从文件导入工作流"""
        # 选择导入文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入工作流", "",
            "JSON 文件 (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            # 导入工作流
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            workflow = Workflow.from_dict(data)
            
            # 保存导入的工作流
            workflow_manager.save_workflow(workflow)
            
            # 加载导入的工作流
            self.current_workflow = workflow
            self.update_workflow_info()
            self.steps_tree.load_workflow(workflow)
            
            QMessageBox.information(
                self, "成功", f"工作流 '{workflow.name}' 已导入。"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "错误", f"导入工作流失败: {str(e)}"
            )
    
    def update_workflow_info(self):
        """更新工作流信息显示"""
        if self.current_workflow is None:
            self.info_label.setText("未打开工作流")
        else:
            self.info_label.setText(
                f"当前工作流: {self.current_workflow.name} ({len(self.current_workflow.steps)} 个步骤)"
            )
    
    def edit_workflow_info(self):
        """编辑工作流信息"""
        if self.current_workflow is None:
            QMessageBox.warning(self, "错误", "没有可编辑的工作流。")
            return
        
        # 弹出对话框编辑名称和描述
        dialog = WorkflowNameDialog(
            self.current_workflow.name,
            self.current_workflow.description,
            parent=self
        )
        
        if not dialog.exec_():
            return
        
        name = dialog.get_name()
        if not name:
            QMessageBox.warning(self, "错误", "工作流名称不能为空。")
            return
        
        description = dialog.get_description()
        
        # 更新工作流信息
        self.current_workflow.name = name
        self.current_workflow.description = description
        
        # 保存修改
        workflow_manager.save_workflow(self.current_workflow)
        
        # 更新UI
        self.update_workflow_info()
    
    def add_step(self):
        """添加步骤"""
        if self.current_workflow is None:
            QMessageBox.warning(self, "错误", "请先创建或打开工作流。")
            return
        
        # 获取所有操作类别和操作
        categories = action_registry.get_categories()
        all_actions = []
        
        for category in categories:
            actions = action_registry.get_actions_in_category(category)
            for action_name in actions:
                all_actions.append(f"{category} - {action_name}")
        
        # 弹出选择对话框
        action_str, ok = QInputDialog.getItem(
            self, "添加步骤", "选择操作:",
            all_actions, 0, False
        )
        
        if not ok or not action_str:
            return
        
        # 解析选择
        action_name = action_str.split(" - ")[1]
        
        # 弹出配置对话框
        dialog = StepConfigDialog(action_name, parent=self)
        if not dialog.exec_():
            return
        
        params = dialog.get_params()
        
        # 添加步骤
        step = WorkflowStep(action_name, params)
        self.current_workflow.add_step(step)
        
        # 保存工作流
        workflow_manager.save_workflow(self.current_workflow)
        
        # 更新UI
        self.update_workflow_info()
        
        # 添加到步骤列表
        item = self.steps_tree.add_step_item(step)
        item.setExpanded(True)
    
    def remove_step(self):
        """移除选中的步骤"""
        if self.current_workflow is None:
            return
        
        # 获取选中的项
        item = self.steps_tree.currentItem()
        if item is None:
            QMessageBox.warning(self, "错误", "请先选择要删除的步骤。")
            return
        
        # 获取项数据
        data = item.data(0, Qt.UserRole)
        if data is None or data.get("type") != "step":
            QMessageBox.warning(self, "错误", "请选择有效的步骤。")
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除", "确定要删除此步骤吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 获取步骤ID
        step_id = data.get("id")
        
        # 从工作流中移除
        if self.current_workflow.remove_step(step_id):
            # 保存工作流
            workflow_manager.save_workflow(self.current_workflow)
            
            # 更新UI
            self.update_workflow_info()
            
            # 从控件中移除
            index = self.steps_tree.indexOfTopLevelItem(item)
            self.steps_tree.takeTopLevelItem(index)
    
    def on_step_edited(self, action: str, data: Dict[str, Any]):
        """处理步骤编辑事件"""
        if self.current_workflow is None:
            return
        
        if action == "add":
            # 添加步骤
            step_data = data.get("step")
            index = data.get("index", -1)
            
            if step_data:
                step = WorkflowStep.from_dict(step_data)
                
                if index >= 0 and index < len(self.current_workflow.steps):
                    # 在指定位置插入
                    self.current_workflow.insert_step(index, step)
                else:
                    # 添加到末尾
                    self.current_workflow.add_step(step)
                
                # 保存工作流
                workflow_manager.save_workflow(self.current_workflow)
                
                # 更新UI
                self.update_workflow_info()
        
        elif action == "update":
            # 更新步骤
            step_id = data.get("id")
            params = data.get("params")
            
            if step_id and params:
                # 更新工作流中的步骤
                self.current_workflow.update_step(step_id, params=params)
                
                # 保存工作流
                workflow_manager.save_workflow(self.current_workflow)
        
        elif action == "delete":
            # 删除步骤
            step_id = data.get("id")
            
            if step_id:
                # 从工作流中移除
                self.current_workflow.remove_step(step_id)
                
                # 保存工作流
                workflow_manager.save_workflow(self.current_workflow)
                
                # 更新UI
                self.update_workflow_info()

        elif action == "reorder":
            ordered_ids = data.get("ordered_ids")
            if ordered_ids and self.current_workflow:
                # 根据新的ID顺序重新排列 self.current_workflow.steps
                
                step_map = {step.id: step for step in self.current_workflow.steps}
                new_steps_list = []
                
                # 验证ID列表是否与当前工作流步骤匹配
                if len(ordered_ids) != len(self.current_workflow.steps):
                    QMessageBox.warning(self, "错误", "步骤重新排序时发生错误：项目数量不匹配。")
                    # 为保持UI和后端一致，可以考虑重新加载工作流以撤销无效的UI更改
                    self.steps_tree.load_workflow(self.current_workflow)
                    return

                valid_reorder = True
                for step_id in ordered_ids:
                    step = step_map.get(step_id)
                    if step:
                        new_steps_list.append(step)
                    else:
                        # 理论上不应发生，因为ID是从UI项中提取的
                        QMessageBox.critical(self, "严重错误", f"步骤重新排序时未找到ID为 {step_id} 的步骤。")
                        self.steps_tree.load_workflow(self.current_workflow) # 恢复UI
                        valid_reorder = False
                        break
                
                if valid_reorder:
                    self.current_workflow.steps = new_steps_list
                    workflow_manager.save_workflow(self.current_workflow)
                    self.update_workflow_info() # 更新如步骤数量等信息
                    # UI树已经由QTreeWidget的内部移动更新，这里不需要重新加载整个树
                # else: 错误已处理或UI已恢复

    
    def get_current_workflow(self) -> Optional[Workflow]:
        """获取当前工作流"""
        return self.current_workflow
    
