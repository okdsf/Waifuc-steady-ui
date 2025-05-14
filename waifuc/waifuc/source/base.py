import copy
from typing import Iterator, Optional, Union

from tqdm.auto import tqdm

from ..action import BaseAction
from ..export import BaseExporter
from ..model import ImageItem
from ..utils import task_ctx, get_task_names, NamedObject


class BaseDataSource:
    def _iter(self) -> Iterator[ImageItem]:
        raise NotImplementedError  # pragma: no cover

    def _iter_from(self) -> Iterator[ImageItem]:
        yield from self._iter()

    def __iter__(self) -> Iterator[ImageItem]:
        yield from self._iter_from()

    def __or__(self, other):
        from .compose import ParallelDataSource
        if isinstance(self, ParallelDataSource):
            if isinstance(other, ParallelDataSource):
                return ParallelDataSource(*self.sources, *other.sources)
            else:
                return ParallelDataSource(*self.sources, other)
        else:
            if isinstance(other, ParallelDataSource):
                return ParallelDataSource(self, *other.sources)
            else:
                return ParallelDataSource(self, other)

    def __add__(self, other):
        from .compose import ComposedDataSource
        if isinstance(self, ComposedDataSource):
            if isinstance(other, ComposedDataSource):
                return ComposedDataSource(*self.sources, *other.sources)
            else:
                return ComposedDataSource(*self.sources, other)
        else:
            if isinstance(other, ComposedDataSource):
                return ComposedDataSource(self, *other.sources)
            else:
                return ComposedDataSource(self, other)

    def __getitem__(self, item):
        from ..action import SliceSelectAction, FirstNSelectAction
        if isinstance(item, slice):
            if item.start is None and item.step is None and item.stop is not None:
                return self.attach(FirstNSelectAction(item.stop))
            else:
                return self.attach(SliceSelectAction(item.start, item.stop, item.step))
        else:
            raise TypeError(f'Data source\'s getitem only accept slices, but {item!r} found.')

    def attach(self, *actions: BaseAction) -> 'AttachedDataSource':
        return AttachedDataSource(self, *actions)

    def export(self, exporter: Union[BaseExporter, str], name: Optional[str] = None):
        if isinstance(exporter, str):
            from ..export import SaveExporter
            exporter = SaveExporter(exporter, no_meta=True)

        exporter = copy.deepcopy(exporter)
        exporter.reset()
        with task_ctx(name):
            return exporter.export_from(iter(self))


class NamedDataSource(BaseDataSource, NamedObject):
    def _iter(self) -> Iterator[ImageItem]:
        raise NotImplementedError  # pragma: no cover

    def _iter_from(self) -> Iterator[ImageItem]:
        names = get_task_names()
        if names:
            desc = f'{self} - {".".join(names)}'
        else:
            desc = f'{self}'
        for item in tqdm(self._iter(), desc=desc):
            yield item


class AttachedDataSource(BaseDataSource):
    def __init__(self, source: BaseDataSource, *actions: BaseAction):
        self.source = source
        self.actions = actions

    def _iter(self) -> Iterator[ImageItem]:
        t = self.source
        for action in self.actions:
            action = copy.deepcopy(action)
            action.reset()
            t = action.iter_from(t)

        yield from t


class EmptySource(BaseDataSource):
    def _iter(self) -> Iterator[ImageItem]:
        yield from []
