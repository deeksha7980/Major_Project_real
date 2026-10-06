/* result.js — Result page behaviour */
document.addEventListener("DOMContentLoaded", () => {
  const statusEl = document.getElementById("status-val");
  if (statusEl) {
    // Still processing — auto-refresh every 4s
    const pid = window.location.pathname.split("/").pop();
    const iv = setInterval(async () => {
      const res = await fetch(`/api/prediction_status/${pid}`);
      const data = await res.json();
      statusEl.textContent = data.status;
      if (data.status === "done" || data.status === "failed") {
        clearInterval(iv);
        location.reload();
      }
    }, 4000);
  }

  // Sync all video players (play one → pause others)
  const videos = document.querySelectorAll("video");
  videos.forEach(v => {
    v.addEventListener("play", () => {
      videos.forEach(other => { if (other !== v) other.pause(); });
    });
  });

  // Filter alert rows
  const alertTable = document.querySelector("#alerts-table");
  if (alertTable) {
    const filterBox = document.createElement("input");
    filterBox.type = "text";
    filterBox.placeholder = "Filter alerts...";
    filterBox.style.cssText = "padding:8px 12px; margin-bottom:10px; background:#161a22; color:#eaeaea; border:1px solid #2b3140; border-radius:6px; width:260px;";
    alertTable.parentElement.insertBefore(filterBox, alertTable);
    filterBox.addEventListener("input", () => {
      const q = filterBox.value.toLowerCase();
      alertTable.querySelectorAll("tbody tr, tr").forEach((r, i) => {
        if (i === 0 && r.querySelector("th")) return;
        r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    });
  }
});