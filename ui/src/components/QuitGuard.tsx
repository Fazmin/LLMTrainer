import { emit, listen } from "@tauri-apps/api/event";
import { Dialog } from "radix-ui";
import { useEffect, useState } from "react";
import { backend, isTauri } from "../backend";
import { Button } from "./Button";

/**
 * Closing the window mid-training would end the run. The backend blocks the close and tells us; this asks what to do.
 * Inactive in the browser preview, where there is no window to close.
 */
export function QuitGuard() {
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let unlisten: (() => void) | undefined;
    void listen("close-requested", () => setOpen(true)).then((u) => {
      unlisten = u;
    });
    return () => unlisten?.();
  }, []);

  const stopAndQuit = async () => {
    setSaving(true);
    try {
      await backend.stopRun(true);
      // Wait for the run to finish saving (up to a minute), then quit.
      const t0 = Date.now();
      while (Date.now() - t0 < 60_000 && (await backend.getActiveRun()) != null) {
        await new Promise((r) => setTimeout(r, 250));
      }
    } catch (e) {
      console.error(e);
    }
    await emit("quit-now");
  };

  return (
    <Dialog.Root open={open} onOpenChange={(o) => !saving && setOpen(o)}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 w-[min(460px,92vw)] -translate-x-1/2 -translate-y-1/2 rounded-[14px] border border-hairline bg-surface p-6 shadow-2xl">
          <Dialog.Title className="m-0 text-lg font-semibold">A training is still running</Dialog.Title>
          <Dialog.Description className="mt-2 text-sm leading-relaxed text-ink-2">
            If you quit now it stops. We can save its progress first, so you can continue from where it got to next time.
          </Dialog.Description>
          <div className="mt-6 flex flex-wrap justify-end gap-2">
            <Dialog.Close asChild>
              <Button disabled={saving}>Keep training</Button>
            </Dialog.Close>
            <Button variant="primary" disabled={saving} onClick={stopAndQuit}>
              {saving ? "Saving your progress…" : "Save and quit"}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
