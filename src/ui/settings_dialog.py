"""
设置对话框模块 - 应用程序设置的配置界面
"""
import os
from typing import Dict, Any
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QDialogButtonBox, QPushButton, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.data.config_manager import config_manager


class GeneralSettingsWidget(QWidget):
    """
    通用设置部件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 基本设置
        basic_group = QGroupBox(self.tr("基本设置"))
        basic_layout = QFormLayout(basic_group)
        
        # 输出目录
        output_dir_layout = QHBoxLayout()
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(config_manager.get("general.output_directory", ""))
        
        output_dir_browse = QPushButton(self.tr("浏览..."))
        output_dir_browse.clicked.connect(self.browse_output_dir)
        
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_dir_browse)
        
        basic_layout.addRow(self.tr("默认输出目录:"), output_dir_layout)
        
        # 临时目录
        temp_dir_layout = QHBoxLayout()
        
        self.temp_dir_edit = QLineEdit()
        self.temp_dir_edit.setText(config_manager.get("general.temp_directory", ""))
        self.temp_dir_edit.setPlaceholderText(self.tr("留空使用系统临时目录"))
        
        temp_dir_browse = QPushButton(self.tr("浏览..."))
        temp_dir_browse.clicked.connect(self.browse_temp_dir)
        
        temp_dir_layout.addWidget(self.temp_dir_edit)
        temp_dir_layout.addWidget(temp_dir_browse)
        
        basic_layout.addRow(self.tr("临时目录:"), temp_dir_layout)
        
        # 日志级别
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        
        log_level = config_manager.get("general.log_level", "INFO")
        index = self.log_level_combo.findText(log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        
        basic_layout.addRow(self.tr("日志级别:"), self.log_level_combo)
        
        layout.addWidget(basic_group)
        
        # UI设置
        ui_group = QGroupBox(self.tr("界面设置"))
        ui_layout = QFormLayout(ui_group)
        
        # 主题
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["default", "dark", "light"])
        
        theme = config_manager.get("ui.theme", "default")
        index = self.theme_combo.findText(theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        ui_layout.addRow(self.tr("主题:"), self.theme_combo)
        
        # 语言
        self.language_combo = QComboBox()
        self.language_combo.addItems(["zh_CN", "en_US"])
        
        language = config_manager.get("ui.language", "zh_CN")
        index = self.language_combo.findText(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        
        ui_layout.addRow(self.tr("语言:"), self.language_combo)
        
        # 显示工具提示
        self.show_tooltips_check = QCheckBox()
        self.show_tooltips_check.setChecked(config_manager.get("ui.show_tooltips", True))
        
        ui_layout.addRow(self.tr("显示工具提示:"), self.show_tooltips_check)
        
        layout.addWidget(ui_group)
        
        # 添加空白占位
        layout.addStretch()
    
    def browse_output_dir(self):
        """浏览选择输出目录"""
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("选择默认输出目录"), self.output_dir_edit.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.output_dir_edit.setText(directory)
    
    def browse_temp_dir(self):
        """浏览选择临时目录"""
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("选择临时目录"), self.temp_dir_edit.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.temp_dir_edit.setText(directory)
    
    def save_settings(self):
        """保存设置"""
        # 基本设置
        config_manager.set("general.output_directory", self.output_dir_edit.text())
        config_manager.set("general.temp_directory", self.temp_dir_edit.text())
        config_manager.set("general.log_level", self.log_level_combo.currentText())
        
        # UI设置
        config_manager.set("ui.theme", self.theme_combo.currentText())
        config_manager.set("ui.language", self.language_combo.currentText())
        config_manager.set("ui.show_tooltips", self.show_tooltips_check.isChecked())

    def retranslateUi(self):
        # 刷新分组框和表单标签
        # 基本设置
        basic_group = self.findChild(QGroupBox, None)
        if basic_group:
            basic_group.setTitle(self.tr("基本设置"))
            basic_layout = basic_group.layout()
            if basic_layout:
                basic_layout.setWidget(0, QFormLayout.LabelRole, QLabel(self.tr("默认输出目录:")))
                basic_layout.setWidget(1, QFormLayout.LabelRole, QLabel(self.tr("临时目录:")))
                basic_layout.setWidget(2, QFormLayout.LabelRole, QLabel(self.tr("日志级别:")))
        # 输出目录浏览按钮
        if hasattr(self, 'output_dir_edit'):
            browse_btn = self.output_dir_edit.parentWidget().findChild(QPushButton)
            if browse_btn:
                browse_btn.setText(self.tr("浏览..."))
        # 临时目录浏览按钮
        if hasattr(self, 'temp_dir_edit'):
            browse_btn = self.temp_dir_edit.parentWidget().findChild(QPushButton)
            if browse_btn:
                browse_btn.setText(self.tr("浏览..."))
            self.temp_dir_edit.setPlaceholderText(self.tr("留空使用系统临时目录"))
        # 日志级别下拉框标签无需翻译（内容为英文等级）
        # UI设置
        ui_group = self.findChildren(QGroupBox)[1] if len(self.findChildren(QGroupBox)) > 1 else None
        if ui_group:
            ui_group.setTitle(self.tr("界面设置"))
            ui_layout = ui_group.layout()
            if ui_layout:
                ui_layout.setWidget(0, QFormLayout.LabelRole, QLabel(self.tr("主题:")))
                ui_layout.setWidget(1, QFormLayout.LabelRole, QLabel(self.tr("语言:")))
                ui_layout.setWidget(2, QFormLayout.LabelRole, QLabel(self.tr("显示工具提示:")))
        # 主题下拉框内容
        if hasattr(self, 'theme_combo'):
            self.theme_combo.setItemText(0, self.tr("默认"))
            self.theme_combo.setItemText(1, self.tr("深色"))
            self.theme_combo.setItemText(2, self.tr("浅色"))
        # 语言下拉框内容
        if hasattr(self, 'language_combo'):
            self.language_combo.setItemText(0, self.tr("简体中文"))
            self.language_combo.setItemText(1, self.tr("English"))
        # 工具提示复选框
        if hasattr(self, 'show_tooltips_check'):
            self.show_tooltips_check.setText("")  # label在左侧，无需文本


class ProcessingSettingsWidget(QWidget):
    """
    处理设置部件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 默认设置
        default_group = QGroupBox(self.tr("默认设置"))
        default_layout = QFormLayout(default_group)
        
        # 默认前缀
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setText(config_manager.get("processing.default_prefix", "output"))
        
        default_layout.addRow(self.tr("默认输出前缀:"), self.prefix_edit)
        
        layout.addWidget(default_group)
        
        # 默认尺寸
        sizes_group = QGroupBox(self.tr("默认尺寸设置"))
        sizes_layout = QFormLayout(sizes_group)
        
        # 获取默认尺寸
        default_sizes = config_manager.get("processing.default_sizes", {
            "1:1": 1024,
            "2:3": 960,
            "3:2": 960
        })
        
        # 正方形 (1:1)
        self.size_1_1_spin = QSpinBox()
        self.size_1_1_spin.setRange(32, 4096)
        self.size_1_1_spin.setValue(default_sizes.get("1:1", 1024))
        
        sizes_layout.addRow(self.tr("正方形 (1:1) 最小尺寸:"), self.size_1_1_spin)
        
        # 纵向 (2:3)
        self.size_2_3_spin = QSpinBox()
        self.size_2_3_spin.setRange(32, 4096)
        self.size_2_3_spin.setValue(default_sizes.get("2:3", 960))
        
        sizes_layout.addRow(self.tr("纵向 (2:3) 最小尺寸:"), self.size_2_3_spin)
        
        # 横向 (3:2)
        self.size_3_2_spin = QSpinBox()
        self.size_3_2_spin.setRange(32, 4096)
        self.size_3_2_spin.setValue(default_sizes.get("3:2", 960))
        
        sizes_layout.addRow(self.tr("横向 (3:2) 最小尺寸:"), self.size_3_2_spin)
        
        layout.addWidget(sizes_group)
        
        # 添加空白占位
        layout.addStretch()
    
    def save_settings(self):
        """保存设置"""
        # 默认设置
        config_manager.set("processing.default_prefix", self.prefix_edit.text())
        
        # 默认尺寸
        config_manager.set("processing.default_sizes", {
            "1:1": self.size_1_1_spin.value(),
            "2:3": self.size_2_3_spin.value(),
            "3:2": self.size_3_2_spin.value()
        })


class SourcesSettingsWidget(QWidget):
    """
    数据源设置部件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # Danbooru设置
        danbooru_group = QGroupBox(self.tr("Danbooru 设置"))
        danbooru_layout = QFormLayout(danbooru_group)
        
        # 默认限制
        self.danbooru_limit_spin = QSpinBox()
        self.danbooru_limit_spin.setRange(1, 1000)
        self.danbooru_limit_spin.setValue(
            config_manager.get("sources.danbooru.default_limit", 100)
        )
        
        danbooru_layout.addRow(self.tr("默认下载数量:"), self.danbooru_limit_spin)
        
        layout.addWidget(danbooru_group)
        
        # Sankaku设置
        sankaku_group = QGroupBox(self.tr("Sankaku Complex 设置"))
        sankaku_layout = QFormLayout(sankaku_group)
        
        # 用户名
        self.sankaku_username_edit = QLineEdit()
        self.sankaku_username_edit.setText(
            config_manager.get("sources.sankaku.username", "")
        )
        
        sankaku_layout.addRow(self.tr("用户名:"), self.sankaku_username_edit)
        
        # 密码
        self.sankaku_password_edit = QLineEdit()
        self.sankaku_password_edit.setEchoMode(QLineEdit.Password)
        self.sankaku_password_edit.setText(
            config_manager.get("sources.sankaku.password", "")
        )
        
        sankaku_layout.addRow(self.tr("密码:"), self.sankaku_password_edit)
        
        # 默认限制
        self.sankaku_limit_spin = QSpinBox()
        self.sankaku_limit_spin.setRange(1, 1000)
        self.sankaku_limit_spin.setValue(
            config_manager.get("sources.sankaku.default_limit", 100)
        )
        
        sankaku_layout.addRow(self.tr("默认下载数量:"), self.sankaku_limit_spin)
        
        layout.addWidget(sankaku_group)
        
        # Pixiv设置
        pixiv_group = QGroupBox(self.tr("Pixiv 设置"))
        pixiv_layout = QFormLayout(pixiv_group)
        
        # 用户名
        self.pixiv_username_edit = QLineEdit()
        self.pixiv_username_edit.setText(
            config_manager.get("sources.pixiv.username", "")
        )
        
        pixiv_layout.addRow(self.tr("用户名:"), self.pixiv_username_edit)
        
        # 密码
        self.pixiv_password_edit = QLineEdit()
        self.pixiv_password_edit.setEchoMode(QLineEdit.Password)
        self.pixiv_password_edit.setText(
            config_manager.get("sources.pixiv.password", "")
        )
        
        pixiv_layout.addRow(self.tr("密码:"), self.pixiv_password_edit)
        
        # 默认限制
        self.pixiv_limit_spin = QSpinBox()
        self.pixiv_limit_spin.setRange(1, 1000)
        self.pixiv_limit_spin.setValue(
            config_manager.get("sources.pixiv.default_limit", 100)
        )
        
        pixiv_layout.addRow(self.tr("默认下载数量:"), self.pixiv_limit_spin)
        
        layout.addWidget(pixiv_group)
        
        # 添加空白占位
        layout.addStretch()
    
    def save_settings(self):
        """保存设置"""
        # Danbooru设置
        config_manager.set("sources.danbooru.default_limit", self.danbooru_limit_spin.value())
        
        # Sankaku设置
        config_manager.set("sources.sankaku.username", self.sankaku_username_edit.text())
        config_manager.set("sources.sankaku.password", self.sankaku_password_edit.text())
        config_manager.set("sources.sankaku.default_limit", self.sankaku_limit_spin.value())
        
        # Pixiv设置
        config_manager.set("sources.pixiv.username", self.pixiv_username_edit.text())
        config_manager.set("sources.pixiv.password", self.pixiv_password_edit.text())
        config_manager.set("sources.pixiv.default_limit", self.pixiv_limit_spin.value())


class SettingsDialog(QDialog):
    """
    设置对话框
    """
    languageChanged = pyqtSignal(str)  # 新增信号
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(self.tr("应用程序设置"))
        self.setMinimumSize(500, 400)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 通用设置
        self.general_settings = GeneralSettingsWidget()
        self.tabs.addTab(self.general_settings, self.tr("通用"))
        
        # 处理设置
        self.processing_settings = ProcessingSettingsWidget()
        self.tabs.addTab(self.processing_settings, self.tr("处理"))
        
        # 数据源设置
        self.sources_settings = SourcesSettingsWidget()
        self.tabs.addTab(self.sources_settings, self.tr("数据源"))
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # 监听语言切换
        self.general_settings.language_combo.currentTextChanged.connect(self.on_language_changed)
    
    def on_language_changed(self, lang_code):
        self.languageChanged.emit(lang_code)
    
    def retranslateUi(self):
        self.setWindowTitle(self.tr("应用程序设置"))
        self.tabs.setTabText(0, self.tr("通用"))
        self.tabs.setTabText(1, self.tr("处理"))
        self.tabs.setTabText(2, self.tr("数据源"))
        # 递归刷新子设置页
        if hasattr(self.general_settings, 'retranslateUi'):
            self.general_settings.retranslateUi()
        if hasattr(self.processing_settings, 'retranslateUi'):
            self.processing_settings.retranslateUi()
        if hasattr(self.sources_settings, 'retranslateUi'):
            self.sources_settings.retranslateUi()
    
    def accept(self):
        """接受更改"""
        try:
            # 保存各页面设置
            self.general_settings.save_settings()
            self.processing_settings.save_settings()
            self.sources_settings.save_settings()
            
            super().accept()
        except Exception as e:
            QMessageBox.warning(self, self.tr("错误"), self.tr(f"保存设置时出错: {str(e)}"))
