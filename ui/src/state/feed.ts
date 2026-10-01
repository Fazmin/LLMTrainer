import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { backend } from "../backend";
import { useLive } from "./live";

/**
 * Connect the ordered live feed once, at startup. Messages are folded into the live store, and the few that change
 * stored history also invalidate the matching queries. After a webview reload the returned snapshot restores the view.
 */
export function useLiveFeed() {
  const qc = useQueryClient();
  useEffect(() => {
    let cancelled = false;
    backend
      .subscribeLive((msg) => {
        useLive.getState().apply(msg);
        if (msg.type === "eval") void qc.invalidateQueries({ queryKey: ["series"] });
        if (msg.type === "state") {
          void qc.invalidateQueries({ queryKey: ["runs"] });
          void qc.invalidateQueries({ queryKey: ["run", msg.runId] });
          void qc.invalidateQueries({ queryKey: ["series"] });
        }
      })
      .then((snapshot) => {
        if (!cancelled) useLive.getState().setSnapshot(snapshot);
      })
      .catch((e) => console.error("could not subscribe to live updates", e));
    return () => {
      cancelled = true;
    };
  }, [qc]);
}
