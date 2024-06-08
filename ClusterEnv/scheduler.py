from attrs import define, Factory
from kubernetes import client, config
from typing import List, Callable, MutableSet
import kopf, logging, asyncio


@define
class Scheduler:
    name: str
    selector: Callable[[List[str]], str]
    apiV1: client.CoreV1Api = Factory(client.CoreV1Api)
    _queue_lock: asyncio.Lock = Factory(asyncio.Lock)
    _queue: MutableSet[str] = Factory(list)

    def __attrs_post_init__(self):
        kopf.on.startup()(self.__load_config)
        kopf.on.create("v1", "pods")(self.schedule)
        kopf.on.resume("v1", "pods")(self.pod_qeueue)
        kopf.run()

    def __load_config(self, **kwargs):
        config.load_kube_config()
        # kopf.configure(logging.INFO)
        logging.info(f"Loadded Config File")

    def __ready_nodes(self) -> List[str]:
        return list(
            filter(
                lambda status: status.status == "True" and status.type == "Ready",
                map(lambda node: node.status.conditions, self.apiV1.list_node().items),
            )
        )

    async def push_pod_qeueue(self, meta: dict, status, **_):
        if status.active is None:
            await self._queue_lock.acquire()
            try: 
                self._queue.add(meta.name)
            finally:
                self._queue_lock.release()

    async def schedule(self, name: str, namespace: str, spec: dict, **kwargs):
        pod_schedule_to_this_scheduler = spec.get("schedulerName") == self.name
        if pod_schedule_to_this_scheduler:
            logging.info("Scheduler Has Been Asked To Schedule {name}")
            ready_nodes = self.__ready_nodes()
            if ready_nodes:
                target_node = self.selector(ready_nodes)
                binding = client.V1Binding(
                    target=client.V1ObjectReference(
                        kind="Node", apiVersion="v1", name=target_node
                    ),
                    metadata=client.V1ObjectMeta(name=name),
                )
                self.v1.create_namespaced_binding(namespace=namespace, body=binding)
                if target_node is not None:
                    await self._queue_lock.acquire()
                    try:
                        await self._queue.remove(name)
                    finally:
                        self._queue_lock.release()


if __name__ == "__main__":
    scheduler = Scheduler(name="foobar", selector=lambda x: x[0])
