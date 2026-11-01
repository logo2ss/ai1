import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Let the user pick the same file again (e.g. after a failed upload): browsers fire no
// "change" event when an <input type=file> keeps its previous value. Clear it after React read it.
document.addEventListener(
  "change",
  (e) => {
    const t = e.target;
    if (t instanceof HTMLInputElement && t.type === "file") setTimeout(() => (t.value = ""), 0);
  },
  true,
);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
