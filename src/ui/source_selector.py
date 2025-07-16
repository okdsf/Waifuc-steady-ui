"""
源选择器模块 - 用于选择图像来源
"""
import os
from typing import Optional, List, Dict, Any, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit, QFileDialog, QSpinBox, QCheckBox, QTabWidget,
    QGroupBox, QMessageBox, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QSettings

from src.tools.sources.source_registry import registry as source_registry
from src.data.config_manager import config_manager


class SourceParamsWidget(QWidget):
    """
    源参数设置部件，根据不同的源类型显示不同的参数设置
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        self.layout = QFormLayout(self)
        
        # 参数控件字典
        self.param_widgets = {}
        
        # 默认显示空白
        self.layout.addRow(QLabel(self.tr("请选择数据源类型")))
    
    def set_source_type(self, source_type: str):
        """
        设置源类型，加载对应的参数控件
        
        Args:
            source_type: 源类型名称
        """
        # 清除现有控件
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.param_widgets = {}
        
        if not source_type:
            self.layout.addRow(QLabel(self.tr("请选择数据源类型")))
            return
        
        try:
            # 获取源参数信息
            params = source_registry.get_source_params(source_type)
            
            # 获取配置中的默认值
            default_values = {}
            if source_type == "LocalSource":
                # 本地源使用最近目录
                recent_dirs = config_manager.get("recent_directories", [])
                if recent_dirs:
                    default_values["directory"] = recent_dirs[0]
            elif source_type in ["DanbooruSource", "SankakuSource", "PixivSource", "YandereSource"]:
                # 网络源使用配置中的默认值
                source_key = source_type.replace("Source", "").lower()
                config_section = config_manager.get(f"sources.{source_key}", {})
                
                for key, value in config_section.items():
                    default_values[key] = value
            
            # 创建参数控件
            if source_type == "LocalSource":
                # 本地源特殊处理
                dir_layout = QHBoxLayout()
                
                dir_edit = QLineEdit()
                dir_edit.setPlaceholderText(self.tr("选择图像目录..."))
                if "directory" in default_values:
                    dir_edit.setText(default_values["directory"])
                
                browse_button = QPushButton(self.tr("浏览..."))
                browse_button.clicked.connect(lambda: self.browse_directory(dir_edit))
                
                dir_layout.addWidget(dir_edit)
                dir_layout.addWidget(browse_button)
                
                self.layout.addRow(self.tr("目录:"), dir_layout)
                self.param_widgets["directory"] = dir_edit
            else:
                # 其他源通用处理
                for param_name, param_value in params.items():
                    # 根据参数类型创建控件
                    if param_name == "tags":
                        # 标签特殊处理
                        tags_edit = QLineEdit()
                        tags_edit.setPlaceholderText(self.tr("输入搜索标签，多个标签用空格分隔"))
                        
                        self.layout.addRow(self.tr("标签:"), tags_edit)
                        self.param_widgets["tags"] = tags_edit
                    elif param_name in ["username", "password"]:
                        # 凭据特殊处理
                        cred_edit = QLineEdit()
                        if param_name == "password":
                            cred_edit.setEchoMode(QLineEdit.Password)
                        
                        if param_name in default_values:
                            cred_edit.setText(default_values[param_name])
                        
                        self.layout.addRow(self.tr(f"{param_name.capitalize()}:"), cred_edit)
                        self.param_widgets[param_name] = cred_edit
                    elif param_name == "limit" and isinstance(param_value, int):
                        # 数量限制特殊处理
                        limit_spin = QSpinBox()
                        limit_spin.setRange(1, 1000)
                        
                        default_limit = default_values.get("default_limit", param_value)
                        limit_spin.setValue(default_limit)
                        
                        self.layout.addRow(self.tr("下载数量:"), limit_spin)
                        self.param_widgets["limit"] = limit_spin
        
        except Exception as e:
            self.layout.addRow(QLabel(self.tr(f"加载参数失败: {str(e)}")))
    
    def browse_directory(self, edit_widget):
        """
        浏览选择目录
        
        Args:
            edit_widget: 目录输入控件
        """
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("选择图像目录"), edit_widget.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            edit_widget.setText(directory)
    
    def get_params(self) -> Dict[str, Any]:
        """
        获取设置的参数
        
        Returns:
            参数字典
        """
        params = {}
        
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QLineEdit):
                value = widget.text().strip()
                if param_name == "tags":
                    # 标签需要转换为列表
                    if value:
                        params[param_name] = value.split()
                    else:
                        params[param_name] = []
                else:
                    params[param_name] = value
            elif isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
        
        return params

    def retranslateUi(self):
        # 刷新所有表单标签和控件
        for i in range(self.layout.rowCount()):
            label_item = self.layout.itemAt(i, QFormLayout.LabelRole)
            if label_item and isinstance(label_item.widget(), QLabel):
                label = label_item.widget()
                text = label.text()
                # 简单处理常见标签
                if "目录" in text:
                    label.setText(self.tr("目录:"))
                elif "标签" in text:
                    label.setText(self.tr("标签:"))
                elif "用户名" in text:
                    label.setText(self.tr("用户名:"))
                elif "密码" in text:
                    label.setText(self.tr("密码:"))
                elif "下载数量" in text:
                    label.setText(self.tr("下载数量:"))
                elif "请选择数据源类型" in text:
                    label.setText(self.tr("请选择数据源类型"))
                elif "加载参数失败" in text:
                    label.setText(self.tr("加载参数失败"))
            field_item = self.layout.itemAt(i, QFormLayout.FieldRole)
            if field_item and isinstance(field_item.widget(), QLineEdit):
                widget = field_item.widget()
                if widget.placeholderText():
                    if "选择图像目录" in widget.placeholderText():
                        widget.setPlaceholderText(self.tr("选择图像目录..."))
                    elif "输入搜索标签" in widget.placeholderText():
                        widget.setPlaceholderText(self.tr("输入搜索标签，多个标签用空格分隔"))
        # 刷新浏览按钮
        for widget in self.findChildren(QPushButton):
            if widget.text() == "浏览...":
                widget.setText(self.tr("浏览..."))


class SavedSourceDialog(QDialog):
    """
    保存的数据源选择对话框
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(self.tr("选择保存的数据源"))
        self.setMinimumSize(400, 300)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 添加列表
        self.sources_list = QListWidget()
        layout.addWidget(self.sources_list)
        
        # 加载保存的数据源
        self.load_saved_sources()
        
        # 添加按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    
    def load_saved_sources(self):
        """加载保存的数据源"""
        self.sources_list.clear()
        self.source_configs = []
        
        # 获取最近使用的数据源
        recent_sources = config_manager.get("recent_sources", [])
        
        for source_config in recent_sources:
            source_type = source_config.get("type")
            source_params = source_config.get("params", {})
            
            # 创建显示文本
            if source_type == "LocalSource":
                directory = source_params.get("directory", "")
                display_text = self.tr(f"本地目录: {directory}")
            else:
                tags = source_params.get("tags", [])
                tags_str = " ".join(tags) if tags else self.tr("<无标签>")
                display_text = self.tr(f"{source_type}: {tags_str}")
            
            # 添加到列表
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, len(self.source_configs))
            self.sources_list.addItem(item)
            
            # 保存配置
            self.source_configs.append(source_config)
    
    def get_selected_source(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        获取选择的数据源
        
        Returns:
            (源类型, 源参数) 元组
        """
        selected_items = self.sources_list.selectedItems()
        if not selected_items:
            return None, None
        
        index = selected_items[0].data(Qt.UserRole)
        if index >= len(self.source_configs):
            return None, None
        
        config = self.source_configs[index]
        return config.get("type"), config.get("params", {})

    def retranslateUi(self):
        self.setWindowTitle(self.tr("选择保存的数据源"))
        self.button_box.button(QDialogButtonBox.Ok).setText(self.tr("确定"))
        self.button_box.button(QDialogButtonBox.Cancel).setText(self.tr("取消"))
        self.load_saved_sources()


class SourceSelectorWidget(QWidget):
    """
    源选择器部件，用于选择图像来源
    """
    source_selected = pyqtSignal(str, dict)  # 源类型, 源参数
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        self.init_ui()
        
        # 当前选择的源类型和参数
        self.current_source_type = None
        self.current_source_params = {}
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 新源选项卡
        self.new_source_tab = QWidget()
        new_source_layout = QVBoxLayout(self.new_source_tab)
        
        # 源类型选择
        self.type_layout = QFormLayout()
        self.source_type_combo = QComboBox()
        self.load_source_types()
        self.type_label = QLabel(self.tr("源类型:"))
        self.type_layout.addRow(self.type_label, self.source_type_combo)
        new_source_layout.addLayout(self.type_layout)
        
        # 参数设置
        self.params_group = QGroupBox(self.tr("参数设置"))
        params_layout = QVBoxLayout(self.params_group)
        
        self.params_widget = SourceParamsWidget()
        params_layout.addWidget(self.params_widget)
        
        new_source_layout.addWidget(self.params_group)
        
        # 选择按钮
        self.select_button = QPushButton(self.tr("选择此源"))
        self.select_button.clicked.connect(self.on_select_source)
        new_source_layout.addWidget(self.select_button)
        
        # 添加到选项卡
        self.tabs.addTab(self.new_source_tab, self.tr("新建源"))
        
        # 保存的源选项卡
        self.saved_source_tab = QWidget()
        saved_source_layout = QVBoxLayout(self.saved_source_tab)
        
        # 保存的源列表
        self.saved_sources_list = QListWidget()
        saved_source_layout.addWidget(self.saved_sources_list)
        
        # 加载保存的源
        self.load_saved_sources()
        
        # 选择按钮
        self.load_button = QPushButton(self.tr("加载选中源"))
        self.load_button.clicked.connect(self.on_load_saved_source)
        saved_source_layout.addWidget(self.load_button)
        
        # 添加到选项卡
        self.tabs.addTab(self.saved_source_tab, self.tr("保存的源"))
        
        # 连接信号
        self.source_type_combo.currentTextChanged.connect(self.on_source_type_changed)
        self.saved_sources_list.itemDoubleClicked.connect(self.on_saved_source_double_clicked)
    
    def load_source_types(self):
        """加载所有可用的源类型"""
        self.source_type_combo.clear()
        
        # 添加空选项
        self.source_type_combo.addItem(self.tr("-- 选择源类型 --"), "")
        
        # 获取所有源类别和源类型
        categories = source_registry.get_categories()
        
        for category in categories:
            # 添加类别分隔
            self.source_type_combo.addItem(self.tr(f"=== {category} ==="))
            self.source_type_combo.setItemData(self.source_type_combo.count() - 1, None)
            
            # 添加该类别下的源类型
            sources = source_registry.get_sources_in_category(category)
            for source_name in sources:
                self.source_type_combo.addItem(source_name, source_name)
    
    def load_saved_sources(self):
        """加载保存的数据源"""
        self.saved_sources_list.clear()
        self.saved_source_configs = []
        
        # 获取最近使用的数据源
        recent_sources = config_manager.get("recent_sources", [])
        
        for source_config in recent_sources:
            source_type = source_config.get("type")
            source_params = source_config.get("params", {})
            
            # 创建显示文本
            if source_type == "LocalSource":
                directory = source_params.get("directory", "")
                display_text = self.tr(f"本地目录: {directory}")
            else:
                tags = source_params.get("tags", [])
                tags_str = " ".join(tags) if tags else self.tr("<无标签>")
                display_text = self.tr(f"{source_type}: {tags_str}")
            
            # 添加到列表
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, len(self.saved_source_configs))
            self.saved_sources_list.addItem(item)
            
            # 保存配置
            self.saved_source_configs.append(source_config)
    
    def on_source_type_changed(self, text):
        """
        源类型改变时的处理
        
        Args:
            text: 当前选择的文本
        """
        # 获取源类型
        source_type = self.source_type_combo.currentData()
        
        # 更新参数控件
        self.params_widget.set_source_type(source_type)
    
    def on_select_source(self):
        """选择当前设置的源"""
        # 获取源类型
        source_type = self.source_type_combo.currentData()
        if not source_type:
            QMessageBox.warning(self, self.tr("错误"), self.tr("请先选择源类型。"))
            return
        
        # 获取参数
        params = self.params_widget.get_params()
        
        # 验证必要参数
        if source_type == "LocalSource":
            directory = params.get("directory", "").strip()
            if not directory:
                QMessageBox.warning(self, self.tr("错误"), self.tr("请选择图像目录。"))
                return
            
            if not os.path.exists(directory):
                QMessageBox.warning(self, self.tr("错误"), self.tr(f"目录不存在: {directory}"))
                return
            
            # 保存到最近使用的目录
            config_manager.add_recent_directory(directory)
        
        elif "tags" in params and not params["tags"]:
            QMessageBox.warning(self, self.tr("错误"), self.tr("请至少输入一个标签。"))
            return
        
        # 保存源配置
        source_config = {
            "type": source_type,
            "params": params
        }
        config_manager.add_recent_source(source_config)
        
        # 更新当前源
        self.current_source_type = source_type
        self.current_source_params = params
        
        # 发送源选择信号
        self.source_selected.emit(source_type, params)
        
        # 刷新保存的源列表
        self.load_saved_sources()
    
    def on_load_saved_source(self):
        """加载选中的保存源"""
        selected_items = self.saved_sources_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, self.tr("错误"), self.tr("请先选择保存的源。"))
            return
        
        index = selected_items[0].data(Qt.UserRole)
        if index >= len(self.saved_source_configs):
            QMessageBox.warning(self, self.tr("错误"), self.tr("源配置无效。"))
            return
        
        # 获取源配置
        config = self.saved_source_configs[index]
        source_type = config.get("type")
        source_params = config.get("params", {})
        
        # 更新当前源
        self.current_source_type = source_type
        self.current_source_params = source_params
        
        # 发送源选择信号
        self.source_selected.emit(source_type, source_params)
        
        # 移动到列表首位
        config_manager.add_recent_source(config)
        
        # 刷新保存的源列表
        self.load_saved_sources()
    
    def on_saved_source_double_clicked(self, item):
        """
        保存的源双击事件处理
        
        Args:
            item: 双击的项
        """
        self.on_load_saved_source()
    
    def get_selected_source(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        获取当前选择的源
        
        Returns:
            (源类型, 源参数) 元组
        """
        return self.current_source_type, self.current_source_params
    
    def select_source_dialog(self) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        显示源选择对话框
        
        Returns:
            (源类型, 源参数) 元组
        """
        dialog = SavedSourceDialog(self)
        if dialog.exec_():
            source_type, source_params = dialog.get_selected_source()
            
            if source_type:
                # 更新当前源
                self.current_source_type = source_type
                self.current_source_params = source_params
                
                # 发送源选择信号
                self.source_selected.emit(source_type, source_params)
                
                return source_type, source_params
        
        return None, None

    def retranslateUi(self):
        # 刷新窗口标题
        if hasattr(self, 'setWindowTitle'):
            self.setWindowTitle(self.tr("数据源选择器"))
        # 刷新选项卡
        self.tabs.setTabText(0, self.tr("新建源"))
        self.tabs.setTabText(1, self.tr("保存的源"))
        # 刷新源类型下拉框
        self.type_label.setText(self.tr("源类型:"))
        self.load_source_types()
        # 刷新参数设置分组框
        self.params_group.setTitle(self.tr("参数设置"))
        # 刷新选择按钮
        self.select_button.setText(self.tr("选择此源"))
        self.load_button.setText(self.tr("加载选中源"))
        # 刷新保存的源列表内容
        self.load_saved_sources()
        # 递归刷新参数设置部件
        if hasattr(self, 'params_widget') and hasattr(self.params_widget, 'retranslateUi'):
            self.params_widget.retranslateUi()
