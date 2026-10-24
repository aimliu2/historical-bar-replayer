(() => {
  const defaults = window.barReplayerDefaults || {};
  const CHART_INITIAL_VISIBLE_BARS = 150;
  const CHART_PADDING_BARS = 10;
  const symbolSelect = document.getElementById("symbolSelect");
  const modeSelect = document.getElementById("modeSelect");
  const timeframeSelect = document.getElementById("timeframeSelect");
  const timeframeField = document.getElementById("timeframeField");
  const dualPairLabel = document.getElementById("dualPairLabel");
  const clockToggle = document.getElementById("clockToggle");
  const clockButtons = [...document.querySelectorAll(".clock-button")];
  const startTimeInput = document.getElementById("startTime");
  const loadButton = document.getElementById("loadButton");
  const backButton = document.getElementById("backButton");
  const stepButton = document.getElementById("stepButton");
  const playButton = document.getElementById("playButton");
  const resetButton = document.getElementById("resetButton");
  const speedSelect = document.getElementById("speedSelect");
  const status = document.getElementById("status");
  const cursorStatus = document.getElementById("cursorStatus");
  const chartTitle = document.getElementById("chartTitle");
  const chartHint = document.getElementById("chartHint");
  const chartHelpTooltip = document.getElementById("chartHelpTooltip");
  const ohlcReadout = document.getElementById("ohlcReadout");
  const chartElement = document.getElementById("chart");
  const chartTooltip = document.getElementById("chartTooltip");
  const emptyChart = document.getElementById("emptyChart");
  const singleChartPanel = document.getElementById("singleChartPanel");
  const dualChartStack = document.getElementById("dualChartStack");
  const dailyStrip = document.getElementById("dailyStrip");
  const dailyStripCells = document.getElementById("dailyStripCells");
  const dailyStripLabel = document.getElementById("dailyStripLabel");
  const dailyToggleLabel = document.getElementById("dailyToggleLabel");
  const dailyToggle = document.getElementById("dailyToggle");
  const fusionInspector = document.getElementById("fusionInspector");
  const fusionContextData = document.getElementById("fusionContextData");
  const rewindButton = document.getElementById("rewindButton");
  const higherPane = document.getElementById("higherPane");
  const lowerPane = document.getElementById("lowerPane");
  const higherPaneHint = document.getElementById("higherPaneHint");
  const lowerPaneHint = document.getElementById("lowerPaneHint");
  const higherLayerToggles = {
    structure: document.getElementById("higherStructureToggle"),
    orderflow: document.getElementById("higherOrderflowToggle"),
    itr: document.getElementById("higherItrToggle"),
    defense: document.getElementById("higherDefenseToggle"),
    ta: document.getElementById("higherTaToggle"),
    band: document.getElementById("higherBandToggle"),
  };
  const lowerLayerToggles = {
    structure: document.getElementById("lowerStructureToggle"),
    orderflow: document.getElementById("lowerOrderflowToggle"),
    shade: document.getElementById("lowerFlowShadeToggle"),
  };

  let sessionId = null;
  let snapshot = null;
  let playTimer = null;
  let requestActive = false;
  let replayLoadActive = false;
  let chart = null;
  let candleSeries = null;
  let resizeObserver = null;
  let chartBars = [];
  let currentPricePrecision = 5;
  let applyingChartUpdate = false;
  let applyingVisibleRange = false;
  let startTimePicker = null;
  let replayMode = null;
  let activeClock = defaults.dual_clock || "1h";
  let selectedRewindTime = null;
  let pendingClock = null;
  let refreshAfterRequest = false;
  let dualPanes = null;
  let mirroringCrosshair = false;

  function isDualMode(mode) {
    return mode === "dual" || mode === "tri";
  }

  function updateChartHelp(mode) {
    const descriptions = {
      single: "One timeframe · step reveals the next bar; hover to inspect OHLC.",
      dual: "Two lanes share one cursor; each pane shows its own structure and orderflow.",
      tri: "Three timeframes share one cursor; the 1D strip is shown as of known_at.",
    };
    chartHelpTooltip.textContent = descriptions[mode] || descriptions.single;
  }

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle("error", isError);
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
    return payload;
  }

  function setControlsEnabled(enabled) {
    backButton.disabled = !enabled;
    stepButton.disabled = !enabled;
    playButton.disabled = !enabled;
    resetButton.disabled = !enabled;
    updateClockButtons();
  }

  function setReplaySetupLocked(locked) {
    modeSelect.disabled = locked;
    symbolSelect.disabled = locked || symbolSelect.options.length === 0;
    timeframeSelect.disabled = locked || isDualMode(modeSelect.value) || timeframeSelect.options.length === 0;
    startTimeInput.disabled = locked;
    loadButton.disabled = locked || !symbolSelect.value || timeframeSelect.options.length === 0 || (modeSelect.value === "single" && !timeframeSelect.value);
    updateModeView();
  }

  function updateModeView() {
    const dual = isDualMode(modeSelect.value);
    const tri = modeSelect.value === "tri";
    timeframeSelect.classList.toggle("hidden", dual);
    timeframeSelect.disabled = dual || replayLoadActive || timeframeSelect.options.length === 0;
    dualPairLabel.classList.toggle("hidden", !dual);
    dualPairLabel.textContent = tri ? "1D / 1H / 5m" : "1H / 5m";
    clockToggle.classList.toggle("hidden", !dual);
    dailyToggleLabel.classList.toggle("hidden", !tri);
    updateChartHelp(modeSelect.value);
    updateClockButtons();
    updateStartTimeStep();
  }

  function updateClockButtons() {
    for (const button of clockButtons) {
      button.classList.toggle("active", button.dataset.clock === activeClock);
      button.setAttribute("aria-pressed", String(button.dataset.clock === activeClock));
      button.disabled = !sessionId || !isDualMode(replayMode) || replayLoadActive;
    }
    const dualSelected = isDualMode(modeSelect.value) || isDualMode(replayMode);
    higherPane.classList.toggle("clock-active", dualSelected && activeClock === "1h");
    lowerPane.classList.toggle("clock-active", dualSelected && activeClock === "5m");
  }

  function setPlaybackButtonState(playing) {
    const action = playing ? "Pause" : "Play";
    const label = `${action} replay`;
    playButton.dataset.tooltip = label;
    playButton.setAttribute("aria-label", label);
  }

  function formatTimestamp(value) {
    if (!value) return "before first bar";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const utc = date.toISOString();
    return `${utc.slice(0, 16).replace("T", " ")} UTC`;
  }

  function formatPrice(value, precision = currentPricePrecision) {
    return Number(value).toFixed(precision);
  }

  function detectPricePrecision(bars) {
    let decimals = 0;
    for (const bar of bars) {
      for (const value of [bar.open, bar.high, bar.low, bar.close]) {
        const text = String(value).toLowerCase();
        if (text.includes("e-")) {
          decimals = Math.max(decimals, Number(text.split("e-")[1]) || 0);
        } else if (text.includes(".")) {
          decimals = Math.max(decimals, text.split(".")[1].length);
        }
      }
    }
    return Math.max(2, Math.min(8, decimals || 5));
  }

  function updateReadout(bar) {
    ohlcReadout.replaceChildren();
    if (!bar) return;
    for (const [label, value] of [["O", bar.open], ["H", bar.high], ["L", bar.low], ["C", bar.close]]) {
      const item = document.createElement("span");
      item.textContent = `${label} ${formatPrice(value)}`;
      ohlcReadout.append(item);
    }
  }

  function hideChartTooltip(element = chartTooltip) {
    element.style.display = "none";
    element.setAttribute("aria-hidden", "true");
  }

  function showOhlcTooltip(element, container, param, candle) {
    const timestamp = Number(param.time);
    const timeLabel = document.createElement("div");
    timeLabel.className = "chart-tooltip-time";
    timeLabel.textContent = formatTimestamp(new Date(timestamp * 1000).toISOString());

    const values = document.createElement("div");
    values.className = "chart-tooltip-values";
    for (const [label, value] of [
      ["O", candle.open], ["H", candle.high], ["L", candle.low], ["C", candle.close],
    ]) {
      const cell = document.createElement("span");
      cell.textContent = `${label} ${formatPrice(value)}`;
      values.append(cell);
    }
    element.replaceChildren(timeLabel, values);
    element.style.display = "grid";
    element.setAttribute("aria-hidden", "false");

    const margin = 10;
    const width = container.clientWidth;
    const height = container.clientHeight;
    const tipWidth = element.offsetWidth;
    const tipHeight = element.offsetHeight;
    const left = param.point.x + tipWidth + margin > width
      ? param.point.x - tipWidth - margin
      : param.point.x + margin;
    const top = param.point.y + tipHeight + margin > height
      ? param.point.y - tipHeight - margin
      : param.point.y + margin;
    element.style.left = `${Math.min(Math.max(margin, left), Math.max(margin, width - tipWidth - margin))}px`;
    element.style.top = `${Math.min(Math.max(margin, top), Math.max(margin, height - tipHeight - margin))}px`;
  }

  function showChartTooltip(param, candle) {
    showOhlcTooltip(chartTooltip, chartElement, param, candle);
  }

  function showPaneTooltip(state, param) {
    if (
      !param || !param.point || param.time === undefined || param.time === null ||
      param.point.x < 0 || param.point.y < 0 ||
      param.point.x > state.element.clientWidth - 62 ||
      param.point.y > state.element.clientHeight - 25
    ) {
      hideChartTooltip(state.tooltip);
      return;
    }
    const candle = param.seriesData && param.seriesData.get(state.series);
    if (!candle || candle.open === undefined || !Number.isFinite(Number(param.time))) {
      hideChartTooltip(state.tooltip);
      return;
    }
    showOhlcTooltip(state.tooltip, state.element, param, candle);
  }

  function logicalRangeForCount(range, count) {
    if (!range || !count) return null;
    const minimumBars = Math.min(10, count);
    const maximumBars = Math.max(minimumBars, count + CHART_PADDING_BARS);
    let from = Number(range.from);
    let to = Number(range.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) {
      return { from: 0, to: count - 1 + CHART_PADDING_BARS };
    }

    let width = to - from + 1;
    if (width < minimumBars) {
      const center = (from + to) / 2;
      width = minimumBars;
      from = center - (width - 1) / 2;
      to = from + width - 1;
    } else if (width > maximumBars) {
      const center = (from + to) / 2;
      width = maximumBars;
      from = center - (width - 1) / 2;
      to = from + width - 1;
    }
    return { from, to };
  }

  function rangeAfterDataUpdate(oldRange, oldBars, nextBars) {
    if (!nextBars.length) return null;
    const nextPaddedRight = nextBars.length - 1 + CHART_PADDING_BARS;
    if (!oldRange || !oldBars.length) {
      return logicalRangeForCount({
        from: Math.max(0, nextPaddedRight - CHART_INITIAL_VISIBLE_BARS),
        to: nextPaddedRight,
      }, nextBars.length);
    }

    const previousWidth = Math.max(1, oldRange.to - oldRange.from + 1);
    const previousPaddedRight = oldBars.length - 1 + CHART_PADDING_BARS;
    const wasAtRightEdge = Math.abs(oldRange.to - previousPaddedRight) <= 0.5;
    if (wasAtRightEdge) {
      return logicalRangeForCount(
        { from: nextPaddedRight - previousWidth + 1, to: nextPaddedRight },
        nextBars.length,
      );
    }

    const logicalShift = oldBars[0].bar_index - nextBars[0].bar_index;
    return logicalRangeForCount({
      from: oldRange.from + logicalShift,
      to: oldRange.to + logicalShift,
    }, nextBars.length);
  }

  function onCrosshairMove(param) {
    if (!param || !param.point || param.time === undefined || param.time === null) {
      hideChartTooltip();
      return;
    }
    const point = param.point;
    if (
      point.x < 0 || point.y < 0 ||
      point.x > chartElement.clientWidth - 62 ||
      point.y > chartElement.clientHeight - 25
    ) {
      hideChartTooltip();
      return;
    }
    const candle = param.seriesData && param.seriesData.get(candleSeries);
    if (!candle || candle.open === undefined || !Number.isFinite(Number(param.time))) {
      hideChartTooltip();
      return;
    }
    showChartTooltip(param, candle);
  }

  function createChart() {
    if (chart) return;
    if (!window.LightweightCharts) {
      throw new Error("The chart library did not load. Check the connection and reload the page.");
    }
    chart = LightweightCharts.createChart(chartElement, {
      width: chartElement.clientWidth,
      height: chartElement.clientHeight,
      layout: {
        background: { type: "solid", color: "#0b1220" },
        textColor: "#aebbd0",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(117, 139, 171, 0.10)" },
        horzLines: { color: "rgba(117, 139, 171, 0.13)" },
      },
      rightPriceScale: { borderColor: "rgba(117, 139, 171, 0.25)" },
      timeScale: {
        borderColor: "rgba(117, 139, 171, 0.25)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: CHART_PADDING_BARS,
        lockVisibleTimeRangeOnResize: true,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { labelVisible: true, color: "rgba(87, 190, 180, 0.65)" },
        horzLine: { labelVisible: true, color: "rgba(87, 190, 180, 0.65)" },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });
    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#32c7b2",
      downColor: "#fa7185",
      borderVisible: false,
      wickUpColor: "#32c7b2",
      wickDownColor: "#fa7185",
      priceFormat: { type: "price", precision: currentPricePrecision, minMove: 0.00001 },
    });

    chart.subscribeCrosshairMove(onCrosshairMove);
    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || applyingChartUpdate) return;
      const bounded = logicalRangeForCount(range, chartBars.length);
      if (!bounded) return;
      const changed = Math.abs(bounded.from - range.from) > 0.01 || Math.abs(bounded.to - range.to) > 0.01;
      if (changed && !applyingVisibleRange) {
        applyingVisibleRange = true;
        chart.timeScale().setVisibleLogicalRange(bounded);
        applyingVisibleRange = false;
      }
    });

    resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry || !chart) return;
      chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    resizeObserver.observe(chartElement);
  }

  function renderChart(bars) {
    createChart();
    const oldRange = chart.timeScale().getVisibleLogicalRange();
    const oldBars = chartBars;
    const nextBars = [...bars];
    const data = nextBars.map((bar) => ({
      time: Math.floor(new Date(bar.time).getTime() / 1000),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));
    currentPricePrecision = detectPricePrecision(nextBars);
    candleSeries.applyOptions({
      priceFormat: {
        type: "price",
        precision: currentPricePrecision,
        minMove: 1 / (10 ** currentPricePrecision),
      },
    });

    applyingChartUpdate = true;
    candleSeries.setData(data);
    chartBars = nextBars;
    emptyChart.classList.toggle("hidden", nextBars.length > 0);
    hideChartTooltip();
    if (data.length) {
      const range = rangeAfterDataUpdate(oldRange, oldBars, nextBars);
      if (range) chart.timeScale().setVisibleLogicalRange(range);
    }
    applyingChartUpdate = false;
  }

  function paneData(bar) {
    const item = {
      time: Math.floor(new Date(bar.time).getTime() / 1000),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    };
    if (bar.forming) {
      item.color = "#182338";
      item.borderColor = "#90a3bd";
      item.wickColor = "#90a3bd";
    }
    return item;
  }

  function chartTime(value) {
    if (!value) return null;
    const time = Math.floor(new Date(value).getTime() / 1000);
    return Number.isFinite(time) ? time : null;
  }

  function addOverlayLine(state, color, width = 1, lineStyle = LightweightCharts.LineStyle.Solid) {
    return state.chart.addSeries(LightweightCharts.LineSeries, {
      color,
      lineWidth: width,
      lineStyle,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
  }

  function lineValues(rows, valueFor) {
    return (rows || []).flatMap((row) => {
      const time = chartTime(row.time);
      if (time === null) return [];
      const value = valueFor(row);
      return value === null || value === undefined || !Number.isFinite(Number(value))
        ? [{ time }]
        : [{ time, value: Number(value) }];
    });
  }

  function removeTransientLines(state) {
    for (const series of [...state.breakSeries, ...state.levelSeries]) {
      state.chart.removeSeries(series);
    }
    state.breakSeries = [];
    state.levelSeries = [];
  }

  function addHorizontalSegment(state, startValue, endValue, price, color, width, lineStyle) {
    if (!Number.isFinite(Number(price))) return null;
    const visible = state.bars.filter((bar) => !bar.forming);
    if (!visible.length) return null;
    const first = chartTime(visible[0].time);
    const last = chartTime(visible[visible.length - 1].time);
    let start = chartTime(startValue);
    let end = chartTime(endValue);
    if (first === null || last === null || start === null || end === null || end < first || start > last) return null;
    start = Math.max(start, first);
    end = Math.min(end, last);
    if (end <= start) return null;
    const series = addOverlayLine(state, color, width, lineStyle);
    series.setData([{ time: start, value: Number(price) }, { time: end, value: Number(price) }]);
    return series;
  }

  function markerTimeVisible(state, value) {
    const time = chartTime(value);
    const visible = state.bars.filter((bar) => !bar.forming);
    if (time === null || !visible.length) return null;
    const first = chartTime(visible[0].time);
    const last = chartTime(visible[visible.length - 1].time);
    return time >= first && time <= last ? time : null;
  }

  function updateStructureOverlay(state, facts) {
    if (!state?.factSeries) return;
    const isHigher = state.name === "higher";
    const toggles = isHigher ? higherLayerToggles : lowerLayerToggles;
    const rows = facts || {};
    const structureTier = rows.structure_tier || (isHigher ? "str" : "itr");
    const showStructure = Boolean(toggles.structure?.checked);
    const showOrderflow = Boolean(toggles.orderflow?.checked);
    const showItr = isHigher && Boolean(toggles.itr?.checked);
    const showFlowShade = isHigher ? showOrderflow : Boolean(lowerLayerToggles.shade?.checked);
    const showDefense = isHigher && Boolean(higherLayerToggles.defense?.checked);
    const showTa = isHigher && Boolean(higherLayerToggles.ta?.checked);
    const showBand = isHigher && Boolean(higherLayerToggles.band?.checked);
    removeTransientLines(state);
    const flowsByTier = rows.orderflow_by_tier || {};
    const flows = rows.orderflow || flowsByTier.ltr || [];
    const itrFlows = flowsByTier.itr || [];
    const defenses = rows.str_defense || [];
    const taRows = rows.ta || [];
    const flowLine = state.factSeries.orderflowDefended;
    flowLine.setData(lineValues(flows, (row) => row.defended?.price));
    state.factSeries.orderflowR.setData(lineValues(flows, (row) => row.provisional_open?.reaffirm_level));
    state.factSeries.orderflowP.setData(lineValues(flows, (row) => row.provisional_open?.prev_level));
    if (state.factSeries.orderflowItr) {
      state.factSeries.orderflowItr.setData(lineValues(itrFlows, (row) => row.defended?.price));
      state.factSeries.orderflowItr.applyOptions({ visible: showItr });
    }
    if (state.factSeries.strDefense) {
      state.factSeries.strDefense.setData(lineValues(defenses, (row) => row.alive ? row.price : null));
      state.factSeries.strDefense.applyOptions({ visible: showDefense });
    }
    const plottedEma = Number(defaults.ta_plot_1h || 100);
    for (const [span, series] of Object.entries(state.factSeries.ema || {})) {
      series.setData(lineValues(taRows, (row) => row.ema?.[span]));
      series.applyOptions({ visible: showTa && Number(span) === plottedEma });
    }
    for (const [edge, series] of Object.entries(state.factSeries.band || {})) {
      series.setData(lineValues(taRows, (row) => row.band?.[edge]));
      series.applyOptions({ visible: showBand });
    }
    flowLine.applyOptions({ visible: showOrderflow });
    state.factSeries.orderflowR.applyOptions({ visible: showOrderflow });
    state.factSeries.orderflowP.applyOptions({ visible: showOrderflow });

    const structureMarkers = [];
    if (showStructure) {
      for (const pivot of rows.pivots || []) {
        if (pivot.tier !== structureTier) continue;
        const time = markerTimeVisible(state, pivot.pivot_ts);
        if (time === null) continue;
        const high = pivot.side === "high";
        const str = pivot.tier === "str";
        const ltr = pivot.tier === "ltr";
        structureMarkers.push({
          time,
          position: high ? "aboveBar" : "belowBar",
          color: str ? "#9e9e9e" : high ? "#7e57c2" : "#00897b",
          shape: high ? "arrowDown" : "arrowUp",
          text: pivot.relation || "",
          size: str || (!isHigher && !ltr) ? 1 : 2,
        });
      }
      for (const wick of rows.wicks || []) {
        if (wick.lane !== structureTier) continue;
        const time = markerTimeVisible(state, wick.event_ts);
        if (time === null) continue;
        structureMarkers.push({
          time,
          position: wick.direction === "up" ? "aboveBar" : "belowBar",
          color: "#c2185b",
          shape: "circle",
          text: `w${wick.wick_seq}`,
          size: !isHigher && wick.lane === "ltr" ? 2 : 1,
        });
      }
      for (const event of rows.breaks || []) {
        if (event.lane !== structureTier) continue;
        const time = markerTimeVisible(state, event.event_ts);
        if (time === null) continue;
        const continuation = ["isb", "lsb"].includes(event.code);
        const lineWidth = !isHigher && event.lane === "ltr" ? 2 : 1;
        const line = addHorizontalSegment(
          state,
          event.level_ts,
          event.event_ts,
          event.level_price,
          continuation ? "#1a73e8" : "#f29900",
          lineWidth,
          LightweightCharts.LineStyle.Solid,
        );
        if (line) state.breakSeries.push(line);
        structureMarkers.push({
          time,
          position: event.break_direction === "up" ? "aboveBar" : "belowBar",
          color: continuation ? "#1a73e8" : "#f29900",
          shape: "circle",
          text: event.code,
          size: lineWidth,
        });
      }
      const lastClosed = state.bars.filter((bar) => !bar.forming).slice(-1)[0];
      for (const level of rows.levels || []) {
        if (level.lane !== structureTier) continue;
        const lineWidth = !isHigher && level.lane === "ltr" ? 2 : 1;
        if (level.fate === "active" && lastClosed) {
          const line = addHorizontalSegment(
            state,
            level.pivot_ts,
            lastClosed.time,
            level.price,
            "#546e7a",
            lineWidth,
            LightweightCharts.LineStyle.Dashed,
          );
          if (line) state.levelSeries.push(line);
        } else if (level.fate === "superseded") {
          const line = addHorizontalSegment(
            state,
            level.pivot_ts,
            level.fate_ts,
            level.price,
            "#8d8d88",
            lineWidth,
            LightweightCharts.LineStyle.Dotted,
          );
          if (line) state.levelSeries.push(line);
        }
      }
    }
    structureMarkers.sort((left, right) => left.time - right.time);
    state.factSeries.markers.setMarkers(structureMarkers);
    for (const series of [...state.breakSeries, ...state.levelSeries]) {
      series.applyOptions({ visible: showStructure });
    }
    updateFlowShading(state, flows, showFlowShade);
  }

  function updateFlowShading(state, flowRows, enabled = true) {
    if (!state?.flowShade) return;
    state.flowShade.replaceChildren();
    if (!enabled) return;
    const byTime = new Map((flowRows || []).map((row) => [chartTime(row.time), row]));
    const bars = state.bars.filter((bar) => !bar.forming);
    for (let index = 0; index < bars.length; index += 1) {
      const bar = bars[index];
      const time = chartTime(bar.time);
      const flow = byTime.get(time);
      if (!flow) continue;
      const color = flow.status === "provisional"
        ? "rgba(224, 186, 103, .13)"
        : flow.status === "active" && flow.direction === "bull"
          ? "rgba(46, 125, 50, .10)"
          : flow.status === "active" && flow.direction === "bear"
            ? "rgba(198, 40, 40, .10)"
            : null;
      if (!color) continue;
      const x = state.chart.timeScale().timeToCoordinate(time);
      if (x === null || x === undefined) continue;
      const next = index + 1 < bars.length
        ? state.chart.timeScale().timeToCoordinate(chartTime(bars[index + 1].time))
        : null;
      const previous = index > 0
        ? state.chart.timeScale().timeToCoordinate(chartTime(bars[index - 1].time))
        : null;
      const spacing = Math.abs((next ?? x + Math.abs(x - (previous ?? x - 7))) - x) || 7;
      const shade = document.createElement("span");
      shade.style.left = `${x - spacing / 2}px`;
      shade.style.width = `${Math.max(1, spacing)}px`;
      shade.style.background = color;
      state.flowShade.append(shade);
    }
  }

  function mergeLaneFacts(previous, delta, bars) {
    const base = previous ? { ...previous } : {
      pivots: [], breaks: [], wicks: [], levels: [], orderflow: [], str_defense: [], ta: [],
    };
    if (delta?.structure_tier) base.structure_tier = delta.structure_tier;
    const keyFunctions = {
      pivots: (row) => row.event_id || `${row.tier}|${row.event_ts}|${row.pivot_ts}|${row.price}`,
      breaks: (row) => `${row.lane}|${row.event_ts}|${row.code}|${row.level_ts}`,
      wicks: (row) => `${row.lane}|${row.event_ts}|${row.level_ts}|${row.wick_seq}`,
      levels: (row) => `${row.lane}|${row.side}|${row.pivot_ts}`,
      orderflow: (row) => row.time,
      str_defense: (row) => row.time,
      ta: (row) => row.time,
    };
    for (const [key, keyOf] of Object.entries(keyFunctions)) {
      const byKey = new Map((base[key] || []).map((row) => [keyOf(row), row]));
      for (const row of delta?.[key] || []) byKey.set(keyOf(row), row);
      base[key] = [...byKey.values()];
    }
    base.orderflow_by_tier = { ...(base.orderflow_by_tier || {}) };
    for (const [tier, rows] of Object.entries(delta?.orderflow_by_tier || {})) {
      const byTime = new Map((base.orderflow_by_tier[tier] || []).map((row) => [row.time, row]));
      for (const row of rows || []) byTime.set(row.time, row);
      base.orderflow_by_tier[tier] = [...byTime.values()];
    }
    if (delta?.anchor_restore) base.anchor_restore = delta.anchor_restore;
    const firstClosed = bars.find((bar) => !bar.forming);
    const cutoff = firstClosed ? new Date(firstClosed.time).getTime() : null;
    const timeOf = (value) => value ? new Date(value).getTime() : null;
    base.pivots = base.pivots.filter((row) => cutoff === null || timeOf(row.event_ts) >= cutoff);
    base.breaks = base.breaks.filter((row) => cutoff === null || timeOf(row.event_ts) >= cutoff);
    base.wicks = base.wicks.filter((row) => cutoff === null || timeOf(row.event_ts) >= cutoff);
    base.levels = base.levels.filter((row) => row.fate === "active" || cutoff === null || timeOf(row.fate_ts || row.event_ts) >= cutoff);
    for (const key of ["orderflow", "str_defense", "ta"]) {
      base[key] = base[key].filter((row) => cutoff === null || timeOf(row.time) >= cutoff);
    }
    for (const tier of Object.keys(base.orderflow_by_tier)) {
      base.orderflow_by_tier[tier] = base.orderflow_by_tier[tier].filter((row) => cutoff === null || timeOf(row.time) >= cutoff);
    }
    return base;
  }

  function createDualPane(name) {
    const isHigher = name === "higher";
    const state = {
      name,
      element: document.getElementById(`${name}Chart`),
      tooltip: document.getElementById(`${name}Tooltip`),
      empty: document.getElementById(`${name}Empty`),
      flowShade: document.getElementById(`${name}FlowShade`),
      chart: null,
      series: null,
      resizeObserver: null,
      bars: [],
      totalBars: 0,
      baseBarIndex: null,
      lastBarIndex: null,
      latestTime: null,
      windowBars: 500,
      windowStart: 0,
      applyingRange: false,
      factSeries: null,
      breakSeries: [],
      levelSeries: [],
    };
    state.chart = LightweightCharts.createChart(state.element, {
      width: state.element.clientWidth,
      height: state.element.clientHeight,
      layout: {
        background: { type: "solid", color: "#0b1220" },
        textColor: "#aebbd0",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(117, 139, 171, 0.10)" },
        horzLines: { color: "rgba(117, 139, 171, 0.13)" },
      },
      rightPriceScale: { borderColor: "rgba(117, 139, 171, 0.25)" },
      timeScale: {
        borderColor: "rgba(117, 139, 171, 0.25)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: CHART_PADDING_BARS,
        lockVisibleTimeRangeOnResize: true,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { labelVisible: true, color: "rgba(87, 190, 180, 0.65)" },
        horzLine: { labelVisible: true, color: "rgba(87, 190, 180, 0.65)" },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    state.series = state.chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#32c7b2",
      downColor: "#fa7185",
      borderVisible: true,
      wickUpColor: "#32c7b2",
      wickDownColor: "#fa7185",
      priceFormat: { type: "price", precision: currentPricePrecision, minMove: 0.00001 },
    });
    state.factSeries = {
      orderflowDefended: addOverlayLine(
        state,
        isHigher ? "#212121" : "#e0c16d",
        2,
        LightweightCharts.LineStyle.Solid,
      ),
      orderflowItr: isHigher ? addOverlayLine(state, "#26a69a", 1, LightweightCharts.LineStyle.Dashed) : null,
      orderflowR: addOverlayLine(state, "#1565c0", 1, LightweightCharts.LineStyle.Dashed),
      orderflowP: addOverlayLine(state, "#6a1b9a", 1, LightweightCharts.LineStyle.Dashed),
      strDefense: isHigher
        ? addOverlayLine(state, "#757575", 1, LightweightCharts.LineStyle.Solid)
        : null,
      ema: isHigher ? {
        5: addOverlayLine(state, "#e53935", 1),
        9: addOverlayLine(state, "#00acc1", 1),
        50: addOverlayLine(state, "#7cb342", 1),
        100: addOverlayLine(state, "#fb8c00", 1),
        200: addOverlayLine(state, "#5e35b1", 1),
      } : {},
      band: isHigher ? {
        upper: addOverlayLine(state, "#9575cd", 1, LightweightCharts.LineStyle.Dashed),
        lower: addOverlayLine(state, "#9575cd", 1, LightweightCharts.LineStyle.Dashed),
      } : null,
      markers: LightweightCharts.createSeriesMarkers(state.series, []),
    };
    for (const series of [state.factSeries.orderflowDefended, state.factSeries.orderflowItr, state.factSeries.strDefense].filter(Boolean)) {
      series.applyOptions({ lineType: LightweightCharts.LineType.WithSteps });
    }
    for (const series of Object.values(state.factSeries.ema)) series.applyOptions({ visible: true });
    state.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range || state.applyingRange) return;
      const bounded = clampPaneRange(state, range);
      if (!bounded) return;
      if (Math.abs(bounded.from - range.from) > 0.01 || Math.abs(bounded.to - range.to) > 0.01) {
        state.applyingRange = true;
        state.chart.timeScale().setVisibleLogicalRange(bounded);
        state.applyingRange = false;
      }
      updateDualDecorations();
    });
    state.chart.subscribeCrosshairMove((param) => {
      if (mirroringCrosshair) hideChartTooltip(state.tooltip);
      else showPaneTooltip(state, param);
      mirrorCrosshair(state, param);
    });
    state.chart.subscribeClick((param) => selectRewindTarget(state, param));
    state.resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      state.chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height });
      updateDualDecorations();
    });
    state.resizeObserver.observe(state.element);
    return state;
  }

  function ensureDualPanes(nextSnapshot) {
    if (!window.LightweightCharts) {
      throw new Error("The chart library did not load. Check the connection and reload the page.");
    }
    if (!dualPanes) {
      dualPanes = { higher: createDualPane("higher"), lower: createDualPane("lower") };
    }
    dualPanes.higher.windowBars = nextSnapshot.window_bars?.["1h"] || defaults.dual_window_bars?.["1h"] || 500;
    dualPanes.lower.windowBars = nextSnapshot.window_bars?.["5m"] || defaults.dual_window_bars?.["5m"] || 1000;
  }

  function clampPaneRange(state, range) {
    if (!state.totalBars) return null;
    const minimumBars = Math.min(10, state.windowBars, state.totalBars);
    const maximumBars = Math.max(minimumBars, state.windowBars + CHART_PADDING_BARS);
    const leftLimit = Math.max(0, state.totalBars - state.bars.length);
    const rightLimit = state.totalBars - 1 + CHART_PADDING_BARS;
    let from = Number(range.from);
    let to = Number(range.to);
    if (!Number.isFinite(from) || !Number.isFinite(to)) {
      return { from: Math.max(leftLimit, rightLimit - CHART_INITIAL_VISIBLE_BARS), to: rightLimit };
    }
    let width = to - from + 1;
    if (width < minimumBars) width = minimumBars;
    if (width > maximumBars) width = maximumBars;
    from = Math.max(leftLimit, Math.min(from, rightLimit - width + 1));
    to = from + width - 1;
    if (to > rightLimit) {
      to = rightLimit;
      from = Math.max(leftLimit, to - width + 1);
    }
    return { from, to };
  }

  function setPaneRange(state, range) {
    const bounded = clampPaneRange(state, range);
    if (!bounded) return;
    state.applyingRange = true;
    state.chart.timeScale().setVisibleLogicalRange(bounded);
    state.applyingRange = false;
  }

  function renderPaneSnapshot(state, bars) {
    const oldRange = state.chart.timeScale().getVisibleLogicalRange();
    const oldBars = state.bars;
    const nextBars = [...bars];
    const data = nextBars.map(paneData);
    state.series.setData(data);
    state.bars = nextBars;
    state.totalBars = nextBars.length;
    state.baseBarIndex = nextBars.length ? nextBars[0].bar_index : null;
    state.lastBarIndex = nextBars.length ? nextBars[nextBars.length - 1].bar_index : null;
    state.latestTime = data.length ? data[data.length - 1].time : null;
    state.windowStart = 0;
    state.empty.classList.toggle("hidden", nextBars.length > 0);
    hideChartTooltip(state.tooltip);
    if (data.length) {
      const range = rangeAfterDataUpdate(oldRange, oldBars, nextBars);
      if (range) setPaneRange(state, range);
      else setPaneRange(state, { from: Math.max(0, data.length + CHART_PADDING_BARS - CHART_INITIAL_VISIBLE_BARS), to: data.length - 1 + CHART_PADDING_BARS });
    }
  }

  function updatePaneDelta(state, lane) {
    const oldRange = state.chart.timeScale().getVisibleLogicalRange();
    const oldRight = state.totalBars - 1 + CHART_PADDING_BARS;
    const followRight = oldRange && Math.abs(oldRange.to - oldRight) <= 0.75;
    const priorWidth = oldRange ? Math.max(1, oldRange.to - oldRange.from + 1) : CHART_INITIAL_VISIBLE_BARS;
    if (lane.drop_left > 0) state.bars.splice(0, lane.drop_left);
    const incoming = [...(lane.bars || [])];
    if (state.name === "higher" && lane.forming) incoming.push(lane.forming);
    for (const bar of incoming) {
      const data = paneData(bar);
      const existing = state.bars.findIndex((item) => item.time === bar.time);
      if (existing >= 0) state.bars[existing] = bar;
      else state.bars.push(bar);
      if (state.latestTime === null || data.time > state.latestTime) {
        state.series.update(data);
        state.totalBars += 1;
        state.latestTime = data.time;
      } else if (data.time === state.latestTime) {
        state.series.update(data);
      }
      state.lastBarIndex = bar.bar_index;
    }
    if (state.bars.length > state.windowBars) {
      state.bars.splice(0, state.bars.length - state.windowBars);
    }
    if (state.baseBarIndex === null && state.bars.length) state.baseBarIndex = state.bars[0].bar_index;
    state.windowStart = Math.max(0, state.totalBars - state.bars.length);
    state.empty.classList.toggle("hidden", state.bars.length > 0);
    if (state.totalBars) {
      const right = state.totalBars - 1 + CHART_PADDING_BARS;
      if (followRight) {
        setPaneRange(state, { from: right - priorWidth + 1, to: right });
      } else if (oldRange) {
        setPaneRange(state, oldRange);
      }
    }
  }

  function reframeLowerOnFocus(focus) {
    const state = dualPanes?.lower;
    if (!state || !focus || !state.bars.length || state.baseBarIndex === null) return;
    const from = new Date(focus.hour_open).getTime();
    const to = new Date(focus.hour_close).getTime();
    const focused = state.bars.filter((bar) => {
      const time = new Date(bar.time).getTime();
      return time > from && time <= to;
    });
    if (!focused.length) return;
    const firstLogical = focused[0].bar_index - state.baseBarIndex;
    const lastLogical = focused[focused.length - 1].bar_index - state.baseBarIndex;
    const context = Number(snapshot?.focus_context_bars ?? defaults.dual_focus_context_bars ?? 36);
    const left = Math.max(state.windowStart, firstLogical - context);
    setPaneRange(state, { from: left, to: lastLogical + CHART_PADDING_BARS });
  }

  function dailyTint(status, scale = 1) {
    if (status === "bullish") return `rgba(46, 125, 50, ${0.20 * scale})`;
    if (status === "bearish") return `rgba(198, 40, 40, ${0.20 * scale})`;
    if (status === "provisional") return `rgba(243, 244, 229, ${0.32 * scale})`;
    return null;
  }

  function shortStage(stage) {
    const status = {
      bullish: "bull",
      bearish: "bear",
      provisional: "prov",
    }[stage?.status] || "none";
    return stage?.exit_path ? `${status}/${stage.exit_path}` : status;
  }

  function dailySummary(row) {
    if (!row) return "1D none";
    const sigmaHat = Number(row.vol_sigma_hat);
    const sigmaPips = Number(row.sigma_hat_pips);
    const sigmaPercent = row.vol_sigma_hat === null || row.vol_sigma_hat === undefined || !Number.isFinite(sigmaHat)
      ? "—"
      : `${(sigmaHat * 100).toFixed(3)}%`;
    const pips = row.sigma_hat_pips === null || row.sigma_hat_pips === undefined || !Number.isFinite(sigmaPips)
      ? "—"
      : `${Math.round(sigmaPips)}p`;
    return `1D yday C ${shortStage(row.stage_c)} · C-1 ${shortStage(row.stage_c1)} · C-2 ${shortStage(row.stage_c2)} · today σ̂ ${sigmaPercent} ~ ${pips}`;
  }

  function updateFusionInspector(context) {
    if (!fusionContextData) return;
    fusionContextData.textContent = context
      ? JSON.stringify(context, null, 2)
      : "No Fusion context available at this cursor.";
  }

  function updateDailyStrip() {
    if (!dualPanes || !snapshot || snapshot.mode !== "tri") {
      dailyStrip.classList.add("hidden");
      dailyStripCells.replaceChildren();
      return;
    }
    dailyStrip.classList.remove("hidden");
    const enabled = Boolean(dailyToggle?.checked);
    dailyStrip.classList.toggle("is-disabled", !enabled);
    dailyStrip.setAttribute("aria-hidden", String(!enabled));
    dailyStripCells.replaceChildren();

    const facts = snapshot.daily_facts || {};
    const asOf = facts.state === "available" ? facts.as_of : null;
    dailyStripLabel.textContent = dailySummary(asOf);
    if (!enabled || !asOf) return;

    const rowsByDay = new Map((facts.rows || []).map((row) => [String(row.bar_open).slice(0, 10), row]));
    const bars = dualPanes.higher.bars;
    const cursorDate = snapshot.cursor_time ? new Date(snapshot.cursor_time).toISOString().slice(0, 10) : null;
    const labelledDays = new Set();
    const toX = (bar) => dualPanes.higher.chart.timeScale().timeToCoordinate(chartTime(bar.time));

    for (let index = 0; index < bars.length; index += 1) {
      const bar = bars[index];
      const time = chartTime(bar.time);
      const x = toX(bar);
      if (time === null || x === null || x === undefined) continue;
      const barOpen = new Date(new Date(bar.time).getTime() - 60 * 60 * 1000);
      const day = barOpen.toISOString().slice(0, 10);
      const ownRow = rowsByDay.get(day);
      const today = day === cursorDate;
      const stage = ownRow?.stage_c || asOf.stage_c;
      const tint = dailyTint(stage?.status, !ownRow && today ? 0.5 : 1);
      if (tint) {
        const nextX = index + 1 < bars.length ? toX(bars[index + 1]) : null;
        const previousX = index > 0 ? toX(bars[index - 1]) : null;
        const spacings = [nextX, previousX]
          .filter((value) => value !== null && value !== undefined)
          .map((value) => Math.abs(value - x))
          .filter((value) => value > 0);
        const spacing = spacings.length
          ? spacings.reduce((sum, value) => sum + value, 0) / spacings.length
          : 7;
        const cell = document.createElement("span");
        cell.className = "daily-strip-cell";
        cell.style.left = `${x - spacing / 2}px`;
        cell.style.width = `${Math.max(1, spacing)}px`;
        cell.style.background = tint;
        dailyStripCells.append(cell);
      }

      if (ownRow?.state_changed && !labelledDays.has(day)) {
        labelledDays.add(day);
        const label = document.createElement("span");
        label.className = "daily-strip-transition";
        label.textContent = `1D ${shortStage(ownRow.stage_c).split("/")[0]}`;
        label.style.left = `${x + 3}px`;
        dailyStripCells.append(label);
      }
    }
  }

  function updateDualDecorations() {
    if (!dualPanes || !snapshot || !isDualMode(snapshot.mode)) return;
    updateFlowShading(
      dualPanes.higher,
      snapshot.higher_facts?.orderflow || [],
      Boolean(higherLayerToggles.orderflow?.checked),
    );
    updateFlowShading(
      dualPanes.lower,
      snapshot.lower_facts?.orderflow || [],
      Boolean(lowerLayerToggles.shade?.checked),
    );
    updateDailyStrip();
  }

  function mirrorCrosshair(source, param) {
    if (!dualPanes || mirroringCrosshair) return;
    const target = dualPanes[source.name === "higher" ? "lower" : "higher"];
    if (!param || param.time === undefined || param.time === null) {
      target.chart.clearCrosshairPosition();
      return;
    }
    const time = typeof param.time === "number"
      ? param.time
      : Math.floor(new Date(`${param.time.year}-${String(param.time.month).padStart(2, "0")}-${String(param.time.day).padStart(2, "0")}T00:00:00Z`).getTime() / 1000);
    const matching = target.bars.find((bar) => Math.floor(new Date(bar.time).getTime() / 1000) === time);
    if (!matching) {
      target.chart.clearCrosshairPosition();
      return;
    }
    mirroringCrosshair = true;
    target.chart.setCrosshairPosition(matching.close, time, target.series);
    window.requestAnimationFrame(() => { mirroringCrosshair = false; });
  }

  function selectRewindTarget(state, param) {
    if (!isDualMode(replayMode) || !param || param.time === undefined || param.time === null) return;
    const time = typeof param.time === "number"
      ? param.time
      : Math.floor(new Date(`${param.time.year}-${String(param.time.month).padStart(2, "0")}-${String(param.time.day).padStart(2, "0")}T00:00:00Z`).getTime() / 1000);
    const bar = state.bars.find((item) => Math.floor(new Date(item.time).getTime() / 1000) === time);
    if (!bar || bar.forming || !snapshot?.cursor_time || new Date(bar.time) > new Date(snapshot.cursor_time)) return;
    selectedRewindTime = bar.time;
    rewindButton.textContent = `Rewind to ${formatTimestamp(selectedRewindTime)}`;
    rewindButton.classList.remove("hidden");
  }

  function updateDualStatus() {
    if (!snapshot || !isDualMode(snapshot.mode)) return;
    const lowerCount = snapshot.revealed?.lower || 0;
    const higherCount = snapshot.revealed?.higher || 0;
    const cursor = formatTimestamp(snapshot.cursor_time);
    cursorStatus.textContent = `${snapshot.clock.toUpperCase()} clock · ${cursor} · +${lowerCount} × 5m / +${higherCount} × 1H${snapshot.done ? " · end of data" : ""}`;
    chartTitle.textContent = snapshot.mode === "tri"
      ? `${snapshot.symbol} · 1D / 1H / 5m`
      : `${snapshot.symbol} · 1H / 5m`;
    chartHint.textContent = snapshot.clock === "1h"
      ? "Choose on 1H, then read the focused hour on 5m."
      : "The 5m clock steps one close at a time; 1H forms in place.";
    if (snapshot.mode === "tri" && snapshot.daily_facts?.state === "none") {
      chartHint.textContent += " No 1D export row is available for this cursor.";
    }
    updateReadout(dualPanes?.lower.bars.filter((bar) => !bar.forming).slice(-1)[0] || dualPanes?.higher.bars.filter((bar) => !bar.forming).slice(-1)[0]);
    setControlsEnabled(true);
    stepButton.disabled = snapshot.done;
    for (const button of clockButtons) button.classList.toggle("active", button.dataset.clock === snapshot.clock);
    if (snapshot.done) stopPlayback();
  }

  function applyDualSnapshot(nextSnapshot, { reframeLower = true } = {}) {
    snapshot = nextSnapshot;
    replayMode = snapshot.mode;
    activeClock = snapshot.clock;
    updateChartHelp(replayMode);
    dailyToggleLabel.classList.toggle("hidden", replayMode !== "tri");
    fusionInspector.classList.remove("hidden");
    singleChartPanel.classList.add("hidden");
    dualChartStack.classList.toggle("is-tri", replayMode === "tri");
    dualChartStack.classList.remove("hidden");
    rewindButton.classList.add("hidden");
    selectedRewindTime = null;
    ensureDualPanes(snapshot);
    currentPricePrecision = detectPricePrecision([...(snapshot.higher_bars || []), ...(snapshot.lower_bars || [])]);
    for (const pane of [dualPanes.higher, dualPanes.lower]) {
      pane.series.applyOptions({
        priceFormat: {
          type: "price",
          precision: currentPricePrecision,
          minMove: 1 / (10 ** currentPricePrecision),
        },
      });
    }
    renderPaneSnapshot(dualPanes.higher, snapshot.higher_bars || []);
    renderPaneSnapshot(dualPanes.lower, snapshot.lower_bars || []);
    hideChartTooltip();
    hideChartTooltip(dualPanes.higher.tooltip);
    hideChartTooltip(dualPanes.lower.tooltip);
    updateStructureOverlay(dualPanes.higher, snapshot.higher_facts);
    updateStructureOverlay(dualPanes.lower, snapshot.lower_facts);
    higherPaneHint.textContent = "";
    lowerPaneHint.textContent = "";
    updateDualStatus();
    updateFusionInspector(snapshot.fusion_context);
    updateClockButtons();
    if (reframeLower && snapshot.clock === "1h") reframeLowerOnFocus(snapshot.focus);
    updateDualDecorations();
    if (snapshot.done) stopPlayback();
  }

  async function applyDualDelta(delta) {
    if (!snapshot || !isDualMode(snapshot.mode) || delta.base_cursor !== snapshot.cursor_time) {
      const full = await api(`/api/replay/${sessionId}`);
      applySnapshot(full.snapshot);
      return snapshot;
    }
    const oldCursor = snapshot.cursor_time;
    snapshot = {
      ...snapshot,
      cursor_time: delta.cursor_time,
      done: delta.done,
      clock: delta.clock,
      revealed: delta.revealed,
      focus: delta.focus,
      fusion_context: delta.fusion_context || snapshot.fusion_context,
    };
    if (delta.higher) updatePaneDelta(dualPanes.higher, delta.higher);
    if (delta.lower) {
      updatePaneDelta(dualPanes.lower, delta.lower);
      lowerPaneHint.textContent = "";
      lowerPaneHint.classList.remove("paused-view");
      if (snapshot.clock === "1h") reframeLowerOnFocus(snapshot.focus);
    } else if (snapshot.clock === "1h") {
      lowerPaneHint.textContent = "Paused view · pause to refresh 5m";
      lowerPaneHint.classList.add("paused-view");
    }
    if (delta.higher?.facts) {
      snapshot.higher_facts = mergeLaneFacts(
        snapshot.higher_facts,
        delta.higher.facts,
        dualPanes.higher.bars,
      );
    }
    if (delta.higher?.daily_facts) snapshot.daily_facts = delta.higher.daily_facts;
    if (delta.lower?.facts) {
      snapshot.lower_facts = mergeLaneFacts(
        snapshot.lower_facts,
        delta.lower.facts,
        dualPanes.lower.bars,
      );
    }
    updateStructureOverlay(dualPanes.higher, snapshot.higher_facts);
    updateStructureOverlay(dualPanes.lower, snapshot.lower_facts);
    if (oldCursor !== snapshot.cursor_time) {
      rewindButton.classList.add("hidden");
      selectedRewindTime = null;
    }
    updateDualStatus();
    updateFusionInspector(snapshot.fusion_context);
    updateDualDecorations();
    return snapshot;
  }

  function applySnapshot(nextSnapshot, options = {}) {
    if (isDualMode(nextSnapshot.mode)) {
      applyDualSnapshot(nextSnapshot, options);
      return;
    }
    snapshot = nextSnapshot;
    replayMode = "single";
    updateChartHelp(replayMode);
    dailyToggleLabel.classList.add("hidden");
    fusionInspector.classList.add("hidden");
    dualChartStack.classList.remove("is-tri");
    hideChartTooltip(dualPanes?.higher.tooltip);
    hideChartTooltip(dualPanes?.lower.tooltip);
    updateDailyStrip();
    singleChartPanel.classList.remove("hidden");
    dualChartStack.classList.add("hidden");
    rewindButton.classList.add("hidden");
    const current = snapshot.current_bar;
    chartTitle.textContent = `${snapshot.symbol} · ${snapshot.timeframe.toUpperCase()}`;
    chartHint.textContent = current
      ? `${snapshot.bars.length} bars through the replay cursor`
      : "Historical bars provide context; Step reveals the selected start bar.";
    cursorStatus.textContent = `Cursor: ${formatTimestamp(snapshot.cursor_time)} · index ${snapshot.cursor_index}${snapshot.done ? " · end of data" : ""}`;
    updateReadout(current);
    renderChart(snapshot.bars || []);
    setControlsEnabled(true);
    stepButton.disabled = snapshot.done;
    if (snapshot.done) stopPlayback();
  }

  async function runAction(path, body) {
    if (!sessionId || requestActive) return null;
    requestActive = true;
    try {
      const result = await api(`/api/replay/${sessionId}/${path}`, {
        method: "POST",
        body: body ? JSON.stringify(body) : "{}",
      });
      if (path === "step" && isDualMode(replayMode)) return await applyDualDelta(result.delta);
      applySnapshot(result.snapshot, { reframeLower: !["back", "reset", "rewind"].includes(path) });
      return result.snapshot;
    } catch (error) {
      stopPlayback();
      setStatus(error.message, true);
      return null;
    } finally {
      requestActive = false;
      settleQueuedActions();
    }
  }

  function settleQueuedActions() {
    if (requestActive) return;
    if (pendingClock !== null) {
      const requestedClock = pendingClock;
      pendingClock = null;
      changeClock(requestedClock);
      return;
    }
    if (refreshAfterRequest) {
      refreshAfterRequest = false;
      refreshDualSnapshot();
    }
  }

  async function refreshDualSnapshot() {
    if (!sessionId || !isDualMode(replayMode)) return;
    if (requestActive) {
      refreshAfterRequest = true;
      return;
    }
    requestActive = true;
    try {
      const result = await api(`/api/replay/${sessionId}`);
      applySnapshot(result.snapshot);
      lowerPaneHint.textContent = "";
      lowerPaneHint.classList.remove("paused-view");
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      requestActive = false;
      settleQueuedActions();
    }
  }

  async function changeClock(clock) {
    if (!sessionId || !isDualMode(replayMode) || !["1h", "5m"].includes(clock)) return;
    if (requestActive) {
      pendingClock = clock;
      return;
    }
    requestActive = true;
    activeClock = clock;
    updateStartTimeStep();
    updateClockButtons();
    try {
      const result = await api(`/api/replay/${sessionId}/clock`, {
        method: "POST",
        body: JSON.stringify({ clock }),
      });
      applySnapshot(result.snapshot);
    } catch (error) {
      activeClock = snapshot?.clock || activeClock;
      updateClockButtons();
      setStatus(error.message, true);
    } finally {
      requestActive = false;
      settleQueuedActions();
    }
  }

  function stopPlayback() {
    if (playTimer !== null) {
      window.clearTimeout(playTimer);
      playTimer = null;
    }
    setPlaybackButtonState(false);
  }

  function playbackStep() {
    if (playTimer === null) return;
    const body = isDualMode(replayMode) && snapshot?.clock === "1h" ? { lanes: ["1h"] } : undefined;
    runAction("step", body).then((next) => {
      if (playTimer === null) return;
      if (!next) {
        if (playTimer !== null) playTimer = window.setTimeout(playbackStep, 40);
        return;
      }
      if (next.done) {
        stopPlayback();
        if (isDualMode(next.mode) && next.clock === "1h") refreshDualSnapshot();
        return;
      }
      playTimer = window.setTimeout(playbackStep, Number(speedSelect.value));
    });
  }

  function startPlayback() {
    if (!sessionId || playTimer !== null || snapshot?.done) return;
    setPlaybackButtonState(true);
    playTimer = window.setTimeout(playbackStep, 0);
  }

  function pausePlayback() {
    const needsLowerRefresh = isDualMode(replayMode)
      && (snapshot?.clock === "1h" || replayMode === "tri");
    stopPlayback();
    if (needsLowerRefresh) refreshDualSnapshot();
  }

  function timeframeMinutes(timeframe) {
    const match = String(timeframe || "").trim().toLowerCase().match(/^(\d+)(m|h|d)$/);
    if (!match) return 1;
    const amount = Number.parseInt(match[1], 10);
    if (!Number.isFinite(amount) || amount <= 0) return 1;
    if (match[2] === "m") return amount;
    if (match[2] === "h") return amount * 60;
    return amount * 24 * 60;
  }

  function updateStartTimeStep() {
    const selectedTimeframe = isDualMode(modeSelect.value) ? activeClock : timeframeSelect.value;
    const timeframeStepMinutes = timeframeMinutes(selectedTimeframe);
    const minuteIncrement = Math.max(1, Math.min(60, timeframeStepMinutes));
    const hourIncrement = timeframeStepMinutes >= 60
      ? Math.max(1, Math.floor(timeframeStepMinutes / 60))
      : 1;
    startTimeInput.step = String(timeframeStepMinutes * 60);
    if (!startTimePicker) return;
    startTimePicker.set("minuteIncrement", minuteIncrement);
    startTimePicker.set("hourIncrement", hourIncrement);
    if (startTimePicker.minuteElement) startTimePicker.minuteElement.step = String(minuteIncrement);
    if (startTimePicker.hourElement) startTimePicker.hourElement.step = String(hourIncrement);
  }

  function initializeStartTimePicker() {
    if (!startTimeInput.value) {
      startTimeInput.value = defaults.start_time_picker || "2026-01-01T00:00";
    }
    if (!window.flatpickr) return;
    startTimePicker = window.flatpickr(startTimeInput, {
      allowInput: false,
      appendTo: document.body,
      dateFormat: "Y-m-d\\TH:i",
      disableMobile: true,
      enableTime: true,
      minuteIncrement: 1,
      time_24hr: true,
      onReady: (_dates, _value, instance) => instance.calendarContainer.classList.add("replay-picker"),
    });
    updateStartTimeStep();
  }

  function getStartTimeIso() {
    const value = startTimeInput.value.trim();
    return value ? `${value}:00Z` : null;
  }

  async function loadTimeframes(preferConfig = false) {
    const symbol = symbolSelect.value;
    if (!symbol) return;
    try {
      const result = await api(`/api/timeframes?symbol=${encodeURIComponent(symbol)}`);
      const timeframes = result.timeframes || [];
      timeframeSelect.replaceChildren(...timeframes.map((timeframe) => new Option(timeframe.toUpperCase(), timeframe)));
      timeframeSelect.disabled = timeframes.length === 0;
      loadButton.disabled = !symbolSelect.value || timeframes.length === 0;
      if (!timeframes.length) {
        setStatus(`No timeframes found for ${symbol}.`, true);
        return;
      }
      const preferred = preferConfig && symbol === defaults.symbol ? defaults.timeframe : null;
      timeframeSelect.value = timeframes.includes(preferred) ? preferred : timeframes[0];
      updateStartTimeStep();
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  async function loadCatalog() {
    try {
      modeSelect.value = defaults.mode || "single";
      activeClock = defaults.dual_clock || "1h";
      updateModeView();
      const result = await api("/api/symbols");
      const symbols = result.symbols || [];
      symbolSelect.replaceChildren(...symbols.map((symbol) => new Option(symbol, symbol)));
      symbolSelect.disabled = symbols.length === 0;
      if (!symbols.length) {
        loadButton.disabled = true;
        setStatus("No matching parquet datasets were found under the data root.", true);
        return;
      }
      symbolSelect.value = symbols.includes(defaults.symbol) ? defaults.symbol : symbols[0];
      await loadTimeframes(true);
      if (timeframeSelect.value) {
        setStatus(`${symbols.length} symbol(s) available.`);
        await createReplay();
      }
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  async function createReplay() {
    if (replayLoadActive) return;
    stopPlayback();
    const symbol = symbolSelect.value;
    const timeframe = timeframeSelect.value;
    const mode = modeSelect.value;
    if (!symbol || (mode === "single" && !timeframe)) {
      setStatus("Choose a symbol and timeframe.", true);
      return;
    }
    replayLoadActive = true;
    setReplaySetupLocked(true);
    try {
      const result = await api("/api/replay", {
        method: "POST",
        body: JSON.stringify({
          mode,
          symbol,
          ...(mode === "single" ? { timeframe } : { clock: activeClock }),
          start_time: getStartTimeIso(),
        }),
      });
      sessionId = result.session_id;
      applySnapshot(result.snapshot);
      setStatus(mode === "tri"
        ? `Loaded ${symbol} 1D / 1H / 5m.`
        : mode === "dual"
          ? `Loaded ${symbol} 1H / 5m.`
          : `Loaded ${symbol} ${timeframe.toUpperCase()}.`);
    } catch (error) {
      setStatus(error.message, true);
      setControlsEnabled(false);
    } finally {
      replayLoadActive = false;
      setReplaySetupLocked(false);
    }
  }

  modeSelect.addEventListener("change", () => {
    updateModeView();
    setReplaySetupLocked(false);
  });
  symbolSelect.addEventListener("change", () => loadTimeframes(false));
  timeframeSelect.addEventListener("change", updateStartTimeStep);
  loadButton.addEventListener("click", createReplay);
  stepButton.addEventListener("click", () => runAction("step"));
  backButton.addEventListener("click", () => runAction("back"));
  resetButton.addEventListener("click", () => {
    stopPlayback();
    runAction("reset");
  });
  playButton.addEventListener("click", () => {
    if (playTimer === null) startPlayback();
    else pausePlayback();
  });
  clockButtons.forEach((button) => {
    button.addEventListener("click", () => changeClock(button.dataset.clock));
  });
  rewindButton.addEventListener("click", () => {
    if (selectedRewindTime) runAction("rewind", { time: selectedRewindTime });
  });
  for (const input of Object.values(higherLayerToggles)) {
    input?.addEventListener("change", () => {
      if (dualPanes?.higher) updateStructureOverlay(dualPanes.higher, snapshot?.higher_facts);
    });
  }
  for (const input of Object.values(lowerLayerToggles)) {
    input?.addEventListener("change", () => {
      if (dualPanes?.lower) updateStructureOverlay(dualPanes.lower, snapshot?.lower_facts);
    });
  }
  dailyToggle.addEventListener("change", updateDailyStrip);
  speedSelect.addEventListener("change", () => {
    if (playTimer !== null) {
      stopPlayback();
      startPlayback();
    }
  });
  window.addEventListener("beforeunload", () => {
    stopPlayback();
    if (resizeObserver) resizeObserver.disconnect();
    if (dualPanes) {
      dualPanes.higher.resizeObserver.disconnect();
      dualPanes.lower.resizeObserver.disconnect();
    }
  });

  setControlsEnabled(false);
  loadButton.disabled = true;
  initializeStartTimePicker();
  updateModeView();
  loadCatalog();
})();
