import { HashRouter, Navigate, Route, Routes } from "react-router";
import { QuitGuard } from "./components/QuitGuard";
import { Sidebar } from "./components/Sidebar";
import { useDatasets, useRuns } from "./state/queries";
import { useLiveFeed } from "./state/feed";
import { Chat } from "./screens/Chat";
import { Compare } from "./screens/Compare";
import { Data } from "./screens/Data";
import { Export } from "./screens/Export";
import { Runs } from "./screens/Runs";
import { Settings } from "./screens/Settings";
import { Setup } from "./screens/Setup";
import { Train } from "./screens/Train";
import { Welcome } from "./screens/Welcome";

/** First launch (no text and no trainings yet) opens the welcome flow; afterwards the app opens on Train. */
function Home() {
  const runs = useRuns();
  const datasets = useDatasets();
  if (runs.isLoading || datasets.isLoading) return null;
  const fresh = (runs.data?.length ?? 0) === 0 && (datasets.data?.length ?? 0) === 0;
  return <Navigate to={fresh ? "/welcome" : (runs.data?.length ?? 0) > 0 ? "/train" : "/setup"} replace />;
}

function Shell() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="min-w-0 flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/data" element={<Data />} />
          <Route path="/setup" element={<Setup />} />
          <Route path="/train" element={<Train />} />
          <Route path="/train/:runId" element={<Train />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/export" element={<Export />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </div>
  );
}

export function App() {
  useLiveFeed();
  return (
    <HashRouter>
      <QuitGuard />
      <Routes>
        {/* The welcome flow is full width, without the sidebar, so a new person sees one clear path. */}
        <Route path="/welcome" element={<div className="h-full overflow-y-auto"><Welcome /></div>} />
        <Route path="*" element={<Shell />} />
      </Routes>
    </HashRouter>
  );
}
