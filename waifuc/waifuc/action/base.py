from typing import Iterator, Iterable, Optional

from tqdm.auto import tqdm

from ..model import ImageItem
from ..utils import get_task_names, NamedObject


class ActionStop(Exception):
    pass


class BaseAction:
    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        raise NotImplementedError  # pragma: no cover

    def iter_from(self, iter_: Iterable[ImageItem]) -> Iterator[ImageItem]:
        for item in iter_:
            try:
                yield from self.iter(item)
            except ActionStop:
                break

    def reset(self):
        raise NotImplementedError  # pragma: no cover


class ProcessAction(BaseAction):
    def process(self, item: ImageItem) -> ImageItem:
        raise NotImplementedError  # pragma: no cover

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        yield self.process(item)

    def reset(self):
        pass

    def __call__(self, item: ImageItem) -> ImageItem:
        return self.process(item)


class FilterAction(BaseAction):
    def check(self, item: ImageItem) -> bool:
        raise NotImplementedError  # pragma: no cover

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        if self.check(item):
            yield item

    def reset(self):
        pass

    def __call__(self, item: ImageItem) -> bool:
        return self.check(item)


class ProgressBarAction(BaseAction, NamedObject):
    def __init__(self, total: Optional[int] = None):
        self.total = total

    def _get_desc(self):
        names = get_task_names()
        if names:
            desc = f'{self} - {".".join(names)}'
        else:
            desc = f'{self}'
        return desc

    def iter_from(self, iter_: Iterable[ImageItem]) -> Iterator[ImageItem]:
        for item in tqdm(BaseAction.iter_from(self, iter_), desc=self._get_desc(), total=self.total):
            yield item

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        raise NotImplementedError  # pragma: no cover

    def reset(self):
        raise NotImplementedError  # pragma: no cover


class TerminalAction(BaseAction):
    """
    一个标记基类，用于表示该 Action 是流水线的“终点”。

    继承自此类的 Action 表明它会自己处理最终的输出保存逻辑，
    并且不会再向 Engine 返回任何 ImageItem 流。
    WorkflowEngine 在遇到此类 Action 作为最后一个步骤时，
    应跳过其默认的最终保存操作。
    """
    # 这个类是空的，只用作类型标记
    pass