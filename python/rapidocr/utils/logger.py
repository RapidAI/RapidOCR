# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging

import colorlog


class Logger:
    def __init__(self, log_level=logging.DEBUG, logger_name=None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s] %(asctime)s [RapidOCR] %(filename)s:%(lineno)d: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)

    def get_log(self):
        return self.logger
