//go:build set3lab

// Copyright 2024 TomTonic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"math"
)

// point is one data point in a line panel.
type point struct{ x, y float64 }

// series is one line in a panel.
type series struct {
	name   string
	color  string
	points []point
}

// chartHeader draws a title and a subtitle and returns the y coordinate where
// the plotting area may begin.
func chartHeader(c *canvas, title, subtitle string) float64 {
	c.text(24, 34, title, 17, colText, "start", "600")
	if subtitle != "" {
		c.text(24, 54, subtitle, 12, colMuted, "start", "400")
		return 74
	}
	return 54
}

// legendRow draws a horizontal legend of coloured swatches at the given y.
func legendRow(c *canvas, x, y float64, entries []series) {
	for _, e := range entries {
		c.rect(x, y-8, 11, 11, e.color, "", 0)
		c.text(x+17, y+1, e.name, 12, colText, "start", "400")
		x += 17 + float64(len(e.name))*6.6 + 22
	}
}

// footnote draws the small print at the bottom of a chart.
func footnote(c *canvas, y float64, lines []string) {
	for i, line := range lines {
		c.text(24, y+float64(i)*15, line, 11, colMuted, "start", "400")
	}
}

// speedupChart draws the headline chart: how much faster Set3 is than the
// native map, for every workload and size, with the evidence attached.
//
// Three things are drawn that an ordinary bar chart leaves out, and each of
// them is there because leaving it out is how benchmark charts mislead:
//
//   - the confidence interval, as a whisker, so a wide result cannot be read
//     as a precise one;
//   - the noise floor, as a pale band around zero, being the difference this
//     machine reported between two runs of identical code — a bar that does
//     not leave the band has measured nothing;
//   - unresolved rows in grey rather than in a winner's colour, and drawn
//     rather than dropped.
func speedupChart(rows []runtimeRow, keyType, subtitle string) *canvas {
	rows = filter(rows, func(r runtimeRow) bool { return r.keyType == keyType && !r.skipped })
	if len(rows) == 0 {
		return nil
	}
	scenarios := distinct(rows, func(r runtimeRow) string { return r.scenario })

	const (
		width     = 940
		labelCol  = 250
		rowHeight = 21
		groupGap  = 9
		// Wide enough that the value printed past the end of the longest bar
		// still lands inside the document.
		rightPad = 58
	)
	plotHeight := float64(len(rows))*rowHeight + float64(len(scenarios))*groupGap
	height := 74 + 34 + plotHeight + 68

	c := newCanvas(width, height)
	top := chartHeader(c, "Set3 against map[T]struct{} — key type "+keyType, subtitle)

	lo, hi := 0.0, 0.0
	for _, r := range rows {
		lo = math.Min(lo, math.Min(r.ciLow, r.delta))
		hi = math.Max(hi, math.Max(r.ciHigh, r.delta))
	}
	pad := math.Max(4, (hi-lo)*0.08)
	xs := newLinearScale(lo-pad, hi+pad, labelCol, width-rightPad)

	axisY := top + 20
	plotTop := axisY + 14
	c.rect(labelCol, plotTop, width-rightPad-labelCol, plotHeight, colPanel, "", 0)
	for _, t := range xs.niceTicks(9) {
		x := xs.at(t)
		c.line(x, plotTop, x, plotTop+plotHeight, colGrid, 1, "")
		c.text(x, axisY+4, signedPercent(t, 0), 11, colMuted, "middle", "400")
	}
	zero := xs.at(0)
	c.line(zero, plotTop, zero, plotTop+plotHeight, colAxis, 1.5, "")

	y := plotTop
	for _, scenario := range scenarios {
		group := filter(rows, func(r runtimeRow) bool { return r.scenario == scenario })
		c.text(24, y+14, scenario, 12, colText, "start", "600")
		for _, r := range group {
			drawSpeedupRow(c, r, xs, y, rowHeight, labelCol)
			y += rowHeight
		}
		y += groupGap
	}

	c.line(labelCol, plotTop+plotHeight, width-rightPad, plotTop+plotHeight, colAxis, 1, "")
	c.text((labelCol+width-rightPad)/2, plotTop+plotHeight+26,
		"← native map faster      relative difference      Set3 faster →", 11, colMuted, "middle", "400")
	footnote(c, plotTop+plotHeight+46, []string{
		"Whisker: 95% confidence interval. Pale band: this machine's noise floor, the difference it reported between two runs of identical code.",
		"Grey bars did not clear both zero and that floor — they establish nothing, which is not the same as establishing that the two are equal.",
	})
	return c
}

// drawSpeedupRow draws one workload-and-size row of the speedup chart.
func drawSpeedupRow(c *canvas, r runtimeRow, xs scale, y, rowHeight, labelCol float64) {
	mid := y + rowHeight/2
	c.text(labelCol-12, mid+4, "n = "+shortCount(r.size), 11, colMuted, "end", "400")

	// The noise floor as a band around zero: anything inside it is not a result.
	c.rect(xs.at(-r.noiseFloor), y+2, xs.at(r.noiseFloor)-xs.at(-r.noiseFloor), rowHeight-4, colNoise, "", 0)

	colour := colUnresolved
	if r.resolved {
		colour = colWin
		if r.delta < 0 {
			colour = colLoss
		}
	}
	x0, x1 := xs.at(0), xs.at(r.delta)
	c.rect(math.Min(x0, x1), y+5, math.Abs(x1-x0), rowHeight-10, colour, "", 0)

	// The interval, drawn on top of the bar so a wide one cannot be missed.
	lo, hi := xs.at(r.ciLow), xs.at(r.ciHigh)
	c.line(lo, mid, hi, mid, colText, 1, "")
	c.line(lo, mid-4, lo, mid+4, colText, 1, "")
	c.line(hi, mid-4, hi, mid+4, colText, 1, "")

	anchor, tx := "start", math.Max(x0, x1)+7
	if r.delta < 0 {
		anchor, tx = "end", math.Min(x0, x1)-7
	}
	c.text(tx, mid+4, signedPercent(r.delta, 1), 11, colText, anchor, "600")
}

// costChart draws one panel per workload: cost per item against set size, both
// axes logarithmic, one line per implementation.
//
// The speedup chart says who won; this one says what it cost, which is the
// question that decides whether a 30% difference is worth anything. A workload
// where both sides run in two nanoseconds and one where both take fifty tell
// very different stories about the same percentage.
func costChart(rows []runtimeRow, keyType, subtitle string) *canvas {
	rows = filter(rows, func(r runtimeRow) bool { return r.keyType == keyType && !r.skipped })
	if len(rows) == 0 {
		return nil
	}
	scenarios := distinct(rows, func(r runtimeRow) string { return r.scenario })

	const (
		cols    = 3
		panelW  = 292
		panelH  = 196
		marginX = 24
		gapX    = 10
		gapY    = 34
		headerH = 96
		footerH = 44
	)
	panelRows := (len(scenarios) + cols - 1) / cols
	width := float64(marginX*2 + cols*panelW + (cols-1)*gapX)
	height := headerH + float64(panelRows)*(panelH+gapY) + footerH

	c := newCanvas(width, height)
	chartHeader(c, "Cost per operation — key type "+keyType, subtitle)
	legendRow(c, 24, 78, []series{{name: "Set3", color: colSet3}, {name: "map[T]struct{}", color: colMap}})

	for i, scenario := range scenarios {
		group := filter(rows, func(r runtimeRow) bool { return r.scenario == scenario })
		px := marginX + float64(i%cols)*(panelW+gapX)
		py := headerH + float64(i/cols)*(panelH+gapY)

		var set3Pts, mapPts []point
		unit := ""
		for _, r := range group {
			set3Pts = append(set3Pts, point{float64(r.size), r.set3Ns})
			mapPts = append(mapPts, point{float64(r.size), r.mapNs})
			unit = r.unit
		}
		drawLinePanel(c, px, py, panelW, panelH, scenario, unit,
			[]series{{name: "Set3", color: colSet3, points: set3Pts}, {name: "map", color: colMap, points: mapPts}},
			true)
	}
	footnote(c, height-26, []string{
		"Both axes logarithmic. Each point is the median of a full rtcompare run at that size; see runtime.csv for the interval around it.",
	})
	return c
}

// memoryChart draws the retained footprint per element, one panel per fill
// history.
//
// Retained, not allocated: what it costs to hold the container, measured as
// heap still live after a full collection. The fill history is a separate panel
// rather than a separate line because it is the variable that surprises people
// — the same number of elements costs materially different amounts depending on
// whether the container was given a size hint, grown into, or churned through.
func memoryChart(rows []memoryRow, keyType, subtitle string) *canvas {
	rows = filter(rows, func(r memoryRow) bool { return r.keyType == keyType })
	if len(rows) == 0 {
		return nil
	}
	shapes := distinct(rows, func(r memoryRow) string { return r.shape })

	const (
		cols    = 2
		panelW  = 400
		panelH  = 216
		marginX = 24
		gapX    = 16
		gapY    = 38
		headerH = 96
		footerH = 44
	)
	panelRows := (len(shapes) + cols - 1) / cols
	width := float64(marginX*2 + cols*panelW + (cols-1)*gapX)
	height := headerH + float64(panelRows)*(panelH+gapY) + footerH

	c := newCanvas(width, height)
	chartHeader(c, "Retained memory per element — key type "+keyType, subtitle)
	legendRow(c, 24, 78, []series{{name: "Set3", color: colSet3}, {name: "map[T]struct{}", color: colMap}})

	for i, shape := range shapes {
		group := filter(rows, func(r memoryRow) bool { return r.shape == shape })
		px := marginX + float64(i%cols)*(panelW+gapX)
		py := headerH + float64(i/cols)*(panelH+gapY)

		var set3Pts, mapPts []point
		ratio := 0.0
		for _, r := range group {
			set3Pts = append(set3Pts, point{float64(r.size), r.set3PerElem})
			mapPts = append(mapPts, point{float64(r.size), r.mapPerElem})
			ratio = r.ratio
		}
		drawLinePanel(c, px, py, panelW, panelH, shape, "bytes per element",
			[]series{{name: "Set3", color: colSet3, points: set3Pts}, {name: "map", color: colMap, points: mapPts}},
			false)
		if ratio > 0 {
			c.text(px+panelW-10, py+34, fmt.Sprintf("Set3 uses %.0f%% of the map's bytes", ratio*100), 11, colWin, "end", "600")
		}
	}
	footnote(c, height-26, []string{
		"Heap still live after a full collection, with the container reachable and the key material held constant. The x axis is logarithmic.",
	})
	return c
}

// drawLinePanel draws one framed panel with a titled pair of axes and its
// series. logY selects a logarithmic vertical axis, which the cost panels want
// and the memory panels do not.
func drawLinePanel(c *canvas, x, y, w, h float64, title, unit string, ss []series, logY bool) {
	const (
		padLeft   = 52
		padRight  = 12
		padTop    = 34
		padBottom = 30
	)
	c.rect(x, y, w, h, colPanel, colGrid, 1)
	c.text(x+10, y+20, title, 12, colText, "start", "600")
	if unit != "" {
		c.text(x+w-10, y+20, unit, 10, colMuted, "end", "400")
	}

	minX, maxX := math.Inf(1), math.Inf(-1)
	minY, maxY := math.Inf(1), math.Inf(-1)
	for _, s := range ss {
		for _, p := range s.points {
			minX, maxX = math.Min(minX, p.x), math.Max(maxX, p.x)
			minY, maxY = math.Min(minY, p.y), math.Max(maxY, p.y)
		}
	}
	if math.IsInf(minX, 1) {
		return
	}

	left, right := x+padLeft, x+w-padRight
	top, bottom := y+padTop, y+h-padBottom
	xs := newLogScale(minX, maxX, left, right)

	var ys scale
	if logY {
		ys = newLogScale(minY/1.4, maxY*1.4, bottom, top)
	} else {
		ys = newLinearScale(0, maxY*1.15, bottom, top)
	}

	for _, t := range ys.niceTicks(4) {
		ty := ys.at(t)
		if ty < top-1 || ty > bottom+1 {
			continue
		}
		c.line(left, ty, right, ty, colGrid, 1, "")
		c.text(left-7, ty+4, formatTick(t), 10, colMuted, "end", "400")
	}
	for _, s := range ss {
		for _, p := range s.points {
			tx := xs.at(p.x)
			c.line(tx, bottom, tx, bottom+4, colAxis, 1, "")
			c.text(tx, bottom+17, shortCount(int(p.x)), 10, colMuted, "middle", "400")
		}
		break
	}
	c.line(left, bottom, right, bottom, colAxis, 1, "")

	for _, s := range ss {
		pts := make([][2]float64, 0, len(s.points))
		for _, p := range s.points {
			pts = append(pts, [2]float64{xs.at(p.x), ys.at(p.y)})
		}
		c.polyline(pts, s.color, 2)
		for _, p := range pts {
			c.circle(p[0], p[1], 3, s.color)
		}
	}
}

// formatTick renders an axis value without more digits than it deserves.
func formatTick(v float64) string {
	switch {
	case v >= 1000:
		return shortCount(int(v))
	case v >= 10:
		return fmt.Sprintf("%.0f", v)
	case v >= 1:
		return fmt.Sprintf("%.1f", v)
	default:
		return fmt.Sprintf("%.2f", v)
	}
}
