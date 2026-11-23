// Plan 05 step 1: reads one OPFS file whole through a sync access handle (workers only) and transfers it.
interface SyncHandle {
  read(buffer: BufferSource, options?: { at?: number }): number;
  getSize(): number;
  close(): void;
}

self.onmessage = async (e: MessageEvent<{ dir: string; name: string }>) => {
  try {
    const root = await navigator.storage.getDirectory();
    const file = await (await root.getDirectoryHandle(e.data.dir)).getFileHandle(e.data.name);
    const h = await (file as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle();
    const buffer = new ArrayBuffer(h.getSize());
    h.read(new Uint8Array(buffer), { at: 0 });
    h.close();
    self.postMessage({ buffer }, { transfer: [buffer] });
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};
