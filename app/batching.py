import queue
import threading
import time
import uuid

class BatchRequest:
    def __init__(self,features,deadline=None):
        self.id=str(uuid.uuid4())
        self.features=features
        self.deadline=deadline
        self.result=None
        self.error=None
        self.event=threading.Event()

    def wait(self,timeout=None):
        if not self.event.wait(timeout):
            raise TimeoutError("Batch request timed out.")
        if self.error is not None:
            raise self.error
        return self.result


class DynamicBatcher:
    def __init__(self,process_batch,max_batch_size=16,max_wait_ms=10):
        self.process_batch=process_batch
        self.max_batch_size=max_batch_size
        self.max_wait_ms=max_wait_ms
        self.queue=queue.Queue()
        self.running=False
        self.worker=None

    def start(self):
        if self.running:
            return

        self.running=True
        self.worker=threading.Thread(target=self._run,daemon=True)
        self.worker.start()

    def stop(self):
        self.running=False
        if self.worker is not None:
            self.worker.join(timeout=1)

    def submit(self,features,deadline=None):
        request=BatchRequest(features,deadline)
        self.queue.put(request)
        return request

    def _run(self):
        while self.running:
            try:
                first=self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch=[first]
            start_time=time.monotonic()

            while len(batch)<self.max_batch_size:
                elapsed_ms=(time.monotonic()-start_time)*1000

                if elapsed_ms>=self.max_wait_ms:
                    break

                remaining_ms=self.max_wait_ms-elapsed_ms

                try:
                    request=self.queue.get(timeout=remaining_ms/1000)
                    batch.append(request)
                except queue.Empty:
                    break

            self._process(batch)

    def _process(self,batch):
        active_requests=[]

        for request in batch:
            if request.deadline is not None and time.monotonic()>request.deadline:
                request.error=TimeoutError("Request deadline exceeded.")
                request.event.set()
            else:
                active_requests.append(request)

        if not active_requests:
            return

        try:
            features=[request.features for request in active_requests]
            results=self.process_batch(features)

            if len(results)!=len(active_requests):
                raise RuntimeError("Batch processor returned incorrect number of results.")

            for request,result in zip(active_requests,results):
                request.result=result
                request.event.set()

        except Exception as error:
            for request in active_requests:
                request.error=error
                request.event.set()