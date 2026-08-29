const csvPath = "../predictions.csv";
const region = {
  latMin: 5,
  latMax: 30,
  lonMin: 45,
  lonMax: 105,
};

let rows = [];
let currentDate = "";

const els = {
  pointCount: document.getElementById("pointCount"),
  dateCount: document.getElementById("dateCount"),
  depthCount: document.getElementById("depthCount"),
  datePicker: document.getElementById("datePicker"),
  dateChips: document.getElementById("dateChips"),
  csvInput: document.getElementById("csvInput"),
  selectionLabel: document.getElementById("selectionLabel"),
  sstMetric: document.getElementById("sstMetric"),
  sssMetric: document.getElementById("sssMetric"),
  sshMetric: document.getElementById("sshMetric"),
  windMetric: document.getElementById("windMetric"),
  profileChart: document.getElementById("profileChart"),
  profileRows: document.getElementById("profileRows"),
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = splitCsvLine(lines.shift());
  return lines.map((line) => {
    const values = splitCsvLine(line);
    return headers.reduce((record, header, index) => {
      const raw = values[index] ?? "";
      const numeric = Number(raw);
      record[header] = raw !== "" && Number.isFinite(numeric) ? numeric : raw;
      return record;
    }, {});
  });
}

function splitCsvLine(line) {
  const result = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && quoted && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      result.push(value);
      value = "";
    } else {
      value += char;
    }
  }

  result.push(value);
  return result;
}

function initData(data) {
  rows = data.filter((row) => isInNorthIndianOcean(row));
  const dates = getDates();
  const depthCols = getDepthColumns();

  els.pointCount.textContent = rows.length.toString();
  els.dateCount.textContent = dates.length.toString();
  els.depthCount.textContent = depthCols.length.toString();

  renderDateControls(dates);
  currentDate = dates[0] || "";
  els.datePicker.value = currentDate;
  updateForDate(currentDate);
}

function getDates() {
  return [...new Set(rows.map((row) => row.date).filter(Boolean))].sort();
}

function getDepthColumns() {
  if (!rows.length) return [];
  return Object.keys(rows[0])
    .filter((key) => key.startsWith("pred_T_"))
    .sort((a, b) => depthFromColumn(a) - depthFromColumn(b));
}

function depthFromColumn(column) {
  return Number(column.replace("pred_T_", ""));
}

function renderDateControls(dates) {
  els.dateChips.innerHTML = "";
  if (!dates.length) return;

  els.datePicker.min = dates[0];
  els.datePicker.max = dates[dates.length - 1];

  dates.forEach((date) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "date-chip";
    chip.textContent = formatDate(date);
    chip.dataset.date = date;
    chip.addEventListener("click", () => {
      els.datePicker.value = date;
      updateForDate(date);
    });
    els.dateChips.appendChild(chip);
  });
}

function updateForDate(date) {
  currentDate = date;

  document.querySelectorAll(".date-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.date === date);
  });

  const dayRows = rows.filter((row) => row.date === date);
  renderMap(dayRows);
  resetDetails(dayRows);
}

function renderMap(dayRows) {
  const shallowTemps = dayRows.map((row) => firstPredictedTemperature(row));
  const trace = {
    type: "scattergeo",
    mode: "markers",
    lat: dayRows.map((row) => row.lat),
    lon: dayRows.map((row) => row.lon),
    customdata: dayRows.map((_, index) => index),
    marker: {
      size: 12,
      color: shallowTemps,
      cmin: 22,
      cmax: 31,
      colorscale: [
        [0, "#2563eb"],
        [0.45, "#087f8c"],
        [0.7, "#d9a21b"],
        [1, "#e85d4f"],
      ],
      line: {
        color: "#ffffff",
        width: 1.5,
      },
      colorbar: {
        title: "Temp deg C",
        thickness: 12,
        len: 0.72,
      },
    },
    text: dayRows.map((row) => {
      const temp = firstPredictedTemperature(row);
      return [
        `${row.lat.toFixed(2)}N, ${row.lon.toFixed(2)}E`,
        Number.isFinite(temp) ? `${temp.toFixed(2)} deg C` : "Prediction unavailable",
        formatDate(row.date),
      ].join("<br>");
    }),
    hovertemplate: "%{text}<extra></extra>",
  };
  const labels = {
    type: "scattergeo",
    mode: "text",
    lat: [17, 16, 7.5],
    lon: [62, 88, 76],
    text: ["Arabian Sea", "Bay of Bengal", "North Indian Ocean"],
    textfont: {
      size: 13,
      color: "#19384a",
    },
    hoverinfo: "skip",
  };

  const layout = {
    margin: { t: 8, r: 8, b: 8, l: 8 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    showlegend: false,
    geo: {
      projection: { type: "mercator" },
      showland: true,
      landcolor: "#e5edf0",
      showocean: true,
      oceancolor: "#eaf6f8",
      showcountries: true,
      countrycolor: "#a9bac3",
      coastlinecolor: "#78909c",
      coastlinewidth: 1,
      lataxis: {
        range: [region.latMin, region.latMax],
        showgrid: true,
        gridcolor: "#d8e4ea",
        dtick: 5,
      },
      lonaxis: {
        range: [region.lonMin, region.lonMax],
        showgrid: true,
        gridcolor: "#d8e4ea",
        dtick: 10,
      },
      fitbounds: false,
      resolution: 50,
    },
  };

  const config = {
    displayModeBar: false,
    responsive: true,
    scrollZoom: false,
  };

  Plotly.react("map", dayRows.length ? [trace, labels] : [labels], layout, config);
  const mapNode = document.getElementById("map");
  mapNode.on("plotly_click", (event) => {
    const index = event.points?.[0]?.customdata;
    if (Number.isInteger(index) && dayRows[index]) {
      selectRow(dayRows[index]);
    }
  });
}

function resetDetails(dayRows) {
  els.selectionLabel.textContent = dayRows.length
    ? `${dayRows.length} prediction points available for ${formatDate(currentDate)}.`
    : "No predictions available for this date.";
  els.sstMetric.textContent = "--";
  els.sssMetric.textContent = "--";
  els.sshMetric.textContent = "--";
  els.windMetric.textContent = "--";
  els.profileChart.innerHTML = `<p class="empty">Select a map point to view the reconstructed temperature profile.</p>`;
  els.profileRows.innerHTML = `<tr><td colspan="2">No point selected</td></tr>`;
}

function selectRow(row) {
  els.selectionLabel.textContent = `${formatDate(row.date)} at ${row.lat.toFixed(2)}N, ${row.lon.toFixed(2)}E`;
  els.sstMetric.textContent = metric(row.SST, "deg C");
  els.sssMetric.textContent = metric(row.SSS, "psu");
  els.sshMetric.textContent = metric(row.SSH, "m");
  els.windMetric.textContent = `${vectorSpeed(row.u_wind, row.v_wind).toFixed(2)} m/s`;
  renderProfile(row);
}

function renderProfile(row) {
  const values = getDepthColumns()
    .map((column) => ({
      depth: depthFromColumn(column),
      temp: Number(row[column]),
    }))
    .filter((item) => Number.isFinite(item.temp));

  if (!values.length) {
    els.profileChart.innerHTML = `<p class="empty">No predicted depth values for this point.</p>`;
    els.profileRows.innerHTML = `<tr><td colspan="2">No predicted depth values</td></tr>`;
    return;
  }

  const min = Math.min(...values.map((item) => item.temp));
  const max = Math.max(...values.map((item) => item.temp));
  const span = Math.max(max - min, 0.1);

  els.profileChart.innerHTML = values.map((item) => {
    const width = 12 + ((item.temp - min) / span) * 88;
    return `
      <div class="depth-bar">
        <span>${item.depth} m</span>
        <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
        <strong>${item.temp.toFixed(2)}</strong>
      </div>
    `;
  }).join("");

  els.profileRows.innerHTML = values.map((item) => `
    <tr>
      <td>${item.depth} m</td>
      <td>${item.temp.toFixed(3)} deg C</td>
    </tr>
  `).join("");
}

function firstPredictedTemperature(row) {
  const first = getDepthColumns().find((column) => Number.isFinite(Number(row[column])));
  return first ? Number(row[first]) : NaN;
}

function isInNorthIndianOcean(row) {
  return Number.isFinite(row.lat)
    && Number.isFinite(row.lon)
    && row.lat >= region.latMin
    && row.lat <= region.latMax
    && row.lon >= region.lonMin
    && row.lon <= region.lonMax;
}


function vectorSpeed(u, v) {
  return Math.sqrt((Number(u) || 0) ** 2 + (Number(v) || 0) ** 2);
}

function metric(value, unit) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)} ${unit}` : "--";
}

function formatDate(value) {
  if (!value) return "--";
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00`));
}

els.datePicker.addEventListener("change", (event) => {
  const dates = getDates();
  if (!dates.length) return;

  const requested = event.target.value;
  const date = dates.includes(requested)
    ? requested
    : dates.reduce((closest, candidate) => {
      const candidateDistance = Math.abs(new Date(candidate) - new Date(requested));
      const closestDistance = Math.abs(new Date(closest) - new Date(requested));
      return candidateDistance < closestDistance ? candidate : closest;
    }, dates[0]);

  els.datePicker.value = date;
  updateForDate(date);
});

els.csvInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  initData(parseCsv(await file.text()));
});

fetch(csvPath)
  .then((response) => {
    if (!response.ok) throw new Error(`Could not load ${csvPath}`);
    return response.text();
  })
  .then((text) => initData(parseCsv(text)))
  .catch(() => {
    els.selectionLabel.textContent = "Serve this folder from the repo root or upload predictions.csv.";
    els.profileChart.innerHTML = `<p class="empty">Waiting for prediction data.</p>`;
  });
