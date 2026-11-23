// Plan 05 step 1: writes and reads OPFS files through createSyncAccessHandle, which only workers have.
interface SyncHandle {
  write(buffer: BufferSource, options?: { at?: number }): number;
  read(buffer: BufferSource, options?: { at?: number }): number;
  getSize(): number;
  flush(): void;
  close(): void;
}

self.onmessage = async (e: MessageEvent<{ shards: number; size: number }>) => {
  try {
    const { shards, size } = e.data;
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("probe-sync", { create: true });
    const data = new Uint8Array(size);
    for (let i = 0; i < data.length; i += 4096) data[i] = i & 0xff;
    let t0 = performance.now();
    for (let i = 0; i < shards; i++) {
      const file = await dir.getFileHandle(`s${i}`, { create: true });
      const h = (await (
        file as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }
      ).createSyncAccessHandle()) as SyncHandle;
      h.write(data, { at: 0 });
      h.flush();
      h.close();
    }
    const writeMs = Math.round(performance.now() - t0);
    t0 = performance.now();
    const back = new Uint8Array(size);
    for (let i = 0; i < shards; i++) {
      const file = await dir.getFileHandle(`s${i}`);
      const h = await (file as unknown as { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle();
      h.read(back, { at: 0 });
      h.close();
    }
    const readMs = Math.round(performance.now() - t0);
    await root.removeEntry("probe-sync", { recursive: true });
    self.postMessage({ writeMs, readMs });
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};
