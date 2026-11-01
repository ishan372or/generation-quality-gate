from app.batching import DynamicBatcher

class ModelServerBatcher:
    def __init__(self,process_batch,max_batch_size=16,max_wait_ms=10):
        self.batcher=DynamicBatcher(
            process_batch=process_batch,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        self.batcher.start()

    def submit(self,features,deadline=None):
        request=self.batcher.submit(
            features=features,
            deadline=deadline,
        )
        return request.wait(timeout=30)