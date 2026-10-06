/* ============================================================
   dashboard.js — Prediction History dashboard
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

  const table = document.querySelector(".data-table");
  if (!table) return;

  const tbody = table.querySelector("tbody");
  const rows  = Array.from(tbody.querySelectorAll("tr")).filter(r => r.cells.length > 1);

  /* =========== 1. Auto-refresh pending/running rows =========== */
  const pendingPids = rows
    .filter(r => {
      const badge = r.querySelector(".badge.yellow");
      return badge !== null;
    })
    .map(r => r.cells[0].textContent.replace("#", "").trim());

  if (pendingPids.length > 0) {
    pollStatuses(pendingPids);
    setInterval(() => pollStatuses(pendingPids), 5000);
  }

  async function pollStatuses(pids) {
    for (const pid of pids) {
      try {
        const res = await fetch(`/api/prediction_status/${pid}`);
        if (!res.ok) continue;
        const data = await res.json();
        const row = tbody.querySelector(`tr td:first-child`);
        const targetRow = rows.find(r =>
          r.cells[0].textContent.replace("#", "").trim() === pid
        );
        if (!targetRow) continue;

        // Update status badge
        const statusCell = targetRow.cells[7];
        const cls = data.status === "done" ? "green"
                  : (data.status === "pending" || data.status === "running") ? "yellow"
                  : "red";
        statusCell.innerHTML = `<span class="badge ${cls}">${data.status}</span>`;

        // Update counts
        targetRow.cells[5].textContent = data.total_cows ?? "—";
        targetRow.cells[6].textContent = data.total_alerts ?? "—";

        // Update alert flag
        if (data.alert_flag !== undefined) {
          targetRow.cells[4].innerHTML = data.alert_flag
            ? '<span class="badge red">YES</span>'
            : '<span class="badge green">No</span>';
        }

        // When done, swap in the "View Result" button
        if (data.status === "done" || data.status === "failed") {
          const actionCell = targetRow.cells[targetRow.cells.length - 1];
          const viewBtn = actionCell.querySelector("a.btn-sm");
          if (viewBtn && data.status === "done") {
            viewBtn.textContent = "Result";
            viewBtn.href = `/result/${pid}`;
          }
          // Stop polling this one by removing it from the array
          const idx = pids.indexOf(pid);
          if (idx > -1) pids.splice(idx, 1);
          if (window.showToast) window.showToast(`Prediction #${pid} ${data.status}`, data.status === "done" ? "success" : "danger");
        }
      } catch (e) {
        console.warn("Poll error", e);
      }
    }
  }

  /* =========== 2. Live search box =========== */
  const searchBox = document.createElement("input");
  searchBox.type = "text";
  searchBox.placeholder = "🔍 Search by ID, filename, status...";
  searchBox.className = "dash-search";
  searchBox.style.cssText = `
    width:340px; padding:10px 14px; margin-bottom:14px;
    background:#161a22; border:1px solid #2b3140; border-radius:8px;
    color:#eaeaea; font-size:14px;
  `;
  table.parentElement.insertBefore(searchBox, table);

  searchBox.addEventListener("input", () => {
    const q = searchBox.value.toLowerCase().trim();
    let visible = 0;
    rows.forEach(r => {
      const text = r.textContent.toLowerCase();
      const match = q === "" || text.includes(q);
      r.style.display = match ? "" : "none";
      if (match) visible++;
    });
    updateEmptyRow(visible);
  });

  /* =========== 3. Status filter chips =========== */
  const chips = document.createElement("div");
  chips.style.cssText = "display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap;";
  ["all", "done", "running", "pending", "failed"].forEach(status => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = status.toUpperCase();
    chip.dataset.status = status;
    chip.style.cssText = `
      padding:6px 14px; border-radius:20px; cursor:pointer;
      background:${status === "all" ? "#00e5b0" : "#1c2230"};
      color:${status === "all" ? "#000" : "#eaeaea"};
      border:1px solid #2b3140; font-weight:600; font-size:12px;
      transition:.2s;
    `;
    chip.addEventListener("click", () => {
      document.querySelectorAll(".filter-chip").forEach(c => {
        c.style.background = "#1c2230";
        c.style.color = "#eaeaea";
      });
      chip.style.background = "#00e5b0";
      chip.style.color = "#000";
      filterByStatus(status);
    });
    chip.classList.add("filter-chip");
    chips.appendChild(chip);
  });
  table.parentElement.insertBefore(chips, table);

  function filterByStatus(status) {
    let visible = 0;
    rows.forEach(r => {
      const badge = r.querySelector(".badge");
      const rowStatus = badge ? badge.textContent.toLowerCase() : "";
      const match = status === "all" || rowStatus === status;
      r.style.display = match ? "" : "none";
      if (match) visible++;
    });
    updateEmptyRow(visible);
  }

  /* =========== 4. Empty state =========== */
  function updateEmptyRow(visibleCount) {
    let empty = tbody.querySelector(".empty-filter-row");
    if (visibleCount === 0) {
      if (!empty) {
        empty = document.createElement("tr");
        empty.className = "empty-filter-row";
        empty.innerHTML = `<td colspan="10" style="text-align:center; padding:30px; color:#888">
          No matching predictions.
        </td>`;
        tbody.appendChild(empty);
      }
      empty.style.display = "";
    } else if (empty) {
      empty.style.display = "none";
    }
  }

  /* =========== 5. Sortable columns =========== */
  const headers = table.querySelectorAll("thead th");
  let sortDir = {};
  headers.forEach((th, idx) => {
    if (["Actions", "Video Input", "Output", "Audio"].includes(th.textContent.trim())) return;
    th.style.cursor = "pointer";
    th.style.userSelect = "none";
    th.title = "Click to sort";
    th.innerHTML += ' <span style="opacity:.4">⇅</span>';
    th.addEventListener("click", () => {
      sortDir[idx] = !sortDir[idx];
      const asc = sortDir[idx];
      const visible = rows.filter(r => r.style.display !== "none");
      visible.sort((a, b) => {
        let av = a.cells[idx].textContent.trim();
        let bv = b.cells[idx].textContent.trim();
        const an = parseFloat(av.replace(/[^\d.\-]/g, ""));
        const bn = parseFloat(bv.replace(/[^\d.\-]/g, ""));
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      visible.forEach(r => tbody.appendChild(r));
    });
  });

  /* =========== 6. Bulk select & delete =========== */
  const selectAllCb = document.createElement("input");
  selectAllCb.type = "checkbox";
  selectAllCb.title = "Select all";
  const firstTh = table.querySelector("thead tr");
  const th = document.createElement("th");
  th.appendChild(selectAllCb);
  firstTh.insertBefore(th, firstTh.firstChild);

  const bulkBar = document.createElement("div");
  bulkBar.style.cssText = `
    display:none; position:sticky; bottom:20px; margin-top:16px;
    background:#1c2230; padding:12px 20px; border-radius:10px;
    align-items:center; gap:14px; box-shadow:0 6px 20px rgba(0,0,0,.5);
  `;
  bulkBar.innerHTML = `
    <span class="bulk-count" style="font-weight:600"></span>
    <button class="btn-sm danger bulk-delete">Delete Selected</button>
    <button class="btn-sm bulk-cancel">Cancel</button>
  `;
  table.parentElement.appendChild(bulkBar);

  rows.forEach(r => {
    const cell = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "row-select";
    cell.appendChild(cb);
    r.insertBefore(cell, r.firstChild);
    cb.addEventListener("change", updateBulkBar);
  });

  selectAllCb.addEventListener("change", () => {
    const visible = rows.filter(r => r.style.display !== "none");
    visible.forEach(r => {
      const cb = r.querySelector(".row-select");
      cb.checked = selectAllCb.checked;
    });
    updateBulkBar();
  });

  function updateBulkBar() {
    const checked = rows.filter(r => r.querySelector(".row-select")?.checked);
    if (checked.length > 0) {
      bulkBar.style.display = "flex";
      bulkBar.querySelector(".bulk-count").textContent = `${checked.length} selected`;
    } else {
      bulkBar.style.display = "none";
    }
  }

  bulkBar.querySelector(".bulk-cancel").addEventListener("click", () => {
    rows.forEach(r => { const cb = r.querySelector(".row-select"); if (cb) cb.checked = false; });
    selectAllCb.checked = false;
    updateBulkBar();
  });

  bulkBar.querySelector(".bulk-delete").addEventListener("click", async () => {
    const checked = rows.filter(r => r.querySelector(".row-select")?.checked);
    if (!confirm(`Delete ${checked.length} prediction(s)? This cannot be undone.`)) return;

    const pids = checked.map(r => r.cells[1].textContent.replace("#", "").trim());

    for (const pid of pids) {
      try {
        const res = await fetch(`/prediction/${pid}/delete`, { method: "POST" });
        if (res.ok) {
          const targetRow = rows.find(r =>
            r.cells[1].textContent.replace("#", "").trim() === pid
          );
          if (targetRow) targetRow.remove();
        }
      } catch (e) {
        console.error("Delete failed", pid, e);
      }
    }
    if (window.showToast) window.showToast(`${pids.length} prediction(s) deleted`, "success");
    updateBulkBar();
  });

  /* =========== 7. Export current list to CSV =========== */
  const exportBtn = document.createElement("button");
  exportBtn.type = "button";
  exportBtn.className = "btn-sm";
  exportBtn.style.marginLeft = "12px";
  exportBtn.textContent = "⬇ Export CSV";
  searchBox.parentElement.insertBefore(exportBtn, table);

  exportBtn.addEventListener("click", () => {
    const visRows = rows.filter(r => r.style.display !== "none");
    if (visRows.length === 0) return;

    const headerCells = Array.from(table.querySelectorAll("thead th"))
      .slice(1) // skip checkbox
      .map(th => `"${th.textContent.replace(/⇅/g, "").trim()}"`);

    const csvLines = [headerCells.join(",")];
    visRows.forEach(r => {
      const vals = Array.from(r.cells).slice(1).map(td => {
        const text = td.textContent.replace(/\s+/g, " ").trim();
        return `"${text.replace(/"/g, '""')}"`;
      });
      csvLines.push(vals.join(","));
    });

    const blob = new Blob([csvLines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `predictions_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  /* =========== 8. Row hover cursor =========== */
  rows.forEach(r => {
    r.style.cursor = "pointer";
    r.addEventListener("click", e => {
      if (e.target.closest("input, button, a, form")) return;
      const pid = r.cells[1].textContent.replace("#", "").trim();
      window.location.href = `/prediction/${pid}`;
    });
  });
});