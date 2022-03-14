import numpy as np
import itertools

class ReplayMemory(object):
    def __init__(self, capacity=100000, replace=False, tuple_class=None):
        self.buffer = []
        self.capacity = capacity
        self.replace = replace
        self.tuple_class = tuple_class
        self.fields = tuple_class._fields

    def add(self, record):
        """Any named tuple item."""
        if isinstance(record, self.tuple_class):
            self.buffer.append(record)
        elif isinstance(record, list):
            self.buffer += record

        while self.capacity and self.size > self.capacity:
            self.buffer.pop(0)

    def _reformat(self, indices):
        # Reformat a list of Transition tuples for training.
        # indices: list<int>
        return {
            field_name: np.array([getattr(self.buffer[i], field_name) for i in indices])
            for field_name in self.fields
        }

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=self.replace)
        return self._reformat(idxs)

    def pop(self, batch_size):
        # Pop the first `batch_size` Transition items out.
        i = min(self.size, batch_size)
        batch = self._reformat(range(i))
        self.buffer = self.buffer[i:]
        return batch

    def clean(self):
        self.buffer = []

    def loop(self, batch_size, epoch=None):
        indices = []
        ep = None
        for i in itertools.cycle(range(len(self.buffer))):
            indices.append(i)
            if i == 0:
                ep = 0 if ep is None else ep + 1
            if epoch is not None and ep == epoch:
                break

            if len(indices) == batch_size:
                yield self._reformat(indices)
                indices = []
        if indices:
            yield self._reformat(indices)

    def next_batch(self, batch_size):
        num_batches = (len(self.buffer) + batch_size - 1) // batch_size
        for i in range(num_batches):
            min_idx = batch_size * i
            max_idx = min(len(self.buffer), min_idx+batch_size)
            indices = range(min_idx, max_idx)
            yield self._reformat(indices)

    @property
    def size(self):
        return len(self.buffer)