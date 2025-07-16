#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
# 添加 waifuc-main、src 和 waifuc-main/tools 到 sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.abspath(os.path.dirname(__file__))
tools_dir = os.path.join(base_dir, 'tools')
sys.path.insert(0, base_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, tools_dir)
"""
主应用程序入口 - 启动图像处理工具
"""
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTranslator

# 导入数据层
from data import config_manager

# 导入UI
from ui import MainWindow


def setup_logging():
    """设置日志系统"""
    log_level_str = config_manager.get("general.log_level", "INFO")
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config_manager.config_dir, 'app.log'), 
                'a', 
                'utf-8'
            )
        ]
    )


def main():
    """程序主入口"""
    setup_logging()
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("图像处理工具")
    app.setApplicationVersion("1.0.0")
    
    # 设置风格
    theme = config_manager.get("ui.theme", "default")
    if theme == "dark":
        app.setStyle("Fusion")
        # 设置暗色主题 (如果需要)
        # 实现暗色主题样式...
    elif theme == "light":
        app.setStyle("Fusion")
        # 设置亮色主题 (如果需要)
        # 实现亮色主题样式...
    
    # 加载语言
    lang_code = config_manager.get("ui.language", "zh_CN")
    translator = QTranslator()
    translator.load(f"translations/{lang_code}.qm")
    app.installTranslator(translator)
    
    # 创建主窗口
    main_window = MainWindow()
    
    # 语言切换回调
    def on_language_changed(new_lang):
        config_manager.set("ui.language", new_lang)
        translator.load(f"translations/{new_lang}.qm")
        app.installTranslator(translator)
        main_window.retranslateUi()  # 需要在 MainWindow 实现
    if hasattr(main_window, 'languageChanged'):
        main_window.languageChanged.connect(on_language_changed)
    
    # 运行应用
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()