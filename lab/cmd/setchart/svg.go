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
	"strconv"
	"strings"
)

// A small SVG writer, deliberately hand-rolled.
//
// The alternative was a plotting library, and it was not worth a dependency in
// a repository whose stated policy is to have as few as possible. What these
// charts need is four primitives and two scales; what a plotting library would
// add is a build-tagged experiment directory that stops compiling when that
// library changes.
//
// One design decision is worth stating because it is not obvious: every chart
// paints an opaque background. GitHub renders README images against a light or
// a dark page depending on the reader's setting, and an SVG with a transparent
// background and dark text becomes unreadable on the dark one. Painting the
// card explicitly is what makes one file work in both.

// Palette. The two series colours are chosen to stay distinguishable in
// grayscale and to survive the most common form of colour blindness, which
// rules out the obvious red/green pairing.
const (
	colBackground = "#ffffff"
	colPanel      = "#f7f8fa"
	colGrid       = "#e2e6ea"
	colAxis       = "#8b949e"
	colText       = "#24292f"
	colMuted      = "#6e7781"
	colSet3       = "#1f6f8b" // deep teal
	colMap        = "#d97706" // amber
	colWin        = "#2f7d32" // green, for "Set3 ahead"
	colLoss       = "#b3261e" // red, for "Set3 behind"
	colUnresolved = "#9aa0a6" // grey, for "not established"
	colNoise      = "#dfe3e8" // the noise floor band
	fontFamily    = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
)

// canvas accumulates SVG elements and renders them as one document.
type canvas struct {
	body   strings.Builder
	width  float64
	height float64
}

// newCanvas starts a chart of the given size with its background painted.
func newCanvas(width, height float64) *canvas {
	c := &canvas{width: width, height: height}
	c.rect(0, 0, width, height, colBackground, "", 0)
	return c
}

// String renders the finished document.
func (c *canvas) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s" viewBox="0 0 %s %s" font-family="%s">`,
		num(c.width), num(c.height), num(c.width), num(c.height), fontFamily)
	b.WriteString(c.body.String())
	b.WriteString("</svg>\n")
	return b.String()
}

func (c *canvas) rect(x, y, w, h float64, fill, stroke string, strokeWidth float64) {
	if w <= 0 || h <= 0 {
		return
	}
	fmt.Fprintf(&c.body, `<rect x="%s" y="%s" width="%s" height="%s" fill="%s"`, num(x), num(y), num(w), num(h), fillOrNone(fill))
	if stroke != "" {
		fmt.Fprintf(&c.body, ` stroke="%s" stroke-width="%s"`, stroke, num(strokeWidth))
	}
	c.body.WriteString(`/>`)
}

func (c *canvas) line(x1, y1, x2, y2 float64, stroke string, width float64, dash string) {
	fmt.Fprintf(&c.body, `<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"`,
		num(x1), num(y1), num(x2), num(y2), stroke, num(width))
	if dash != "" {
		fmt.Fprintf(&c.body, ` stroke-dasharray="%s"`, dash)
	}
	c.body.WriteString(`/>`)
}

func (c *canvas) circle(x, y, r float64, fill string) {
	fmt.Fprintf(&c.body, `<circle cx="%s" cy="%s" r="%s" fill="%s"/>`, num(x), num(y), num(r), fill)
}

// polyline draws a series. Points are already in device coordinates.
func (c *canvas) polyline(pts [][2]float64, stroke string, width float64) {
	if len(pts) < 2 {
		return
	}
	var b strings.Builder
	for i, p := range pts {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s,%s", num(p[0]), num(p[1]))
	}
	fmt.Fprintf(&c.body, `<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round" stroke-linecap="round"/>`,
		b.String(), stroke, num(width))
}

// text anchors are "start", "middle" or "end".
func (c *canvas) text(x, y float64, s string, size float64, fill, anchor, weight string) {
	fmt.Fprintf(&c.body, `<text x="%s" y="%s" font-size="%s" fill="%s" text-anchor="%s" font-weight="%s">%s</text>`,
		num(x), num(y), num(size), fill, anchor, weight, escape(s))
}

func fillOrNone(fill string) string {
	if fill == "" {
		return "none"
	}
	return fill
}

// num formats a coordinate compactly: SVG files here run to a few hundred
// kilobytes and trailing zeroes on every number are a third of that.
func num(v float64) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "0"
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func escape(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;")
	return r.Replace(s)
}

// scale maps a data value onto a device coordinate.
type scale struct {
	dataMin, dataMax float64
	pixMin, pixMax   float64
	logarithmic      bool
}

// newLinearScale maps [dataMin, dataMax] onto [pixMin, pixMax].
func newLinearScale(dataMin, dataMax, pixMin, pixMax float64) scale {
	if dataMax <= dataMin {
		dataMax = dataMin + 1
	}
	return scale{dataMin: dataMin, dataMax: dataMax, pixMin: pixMin, pixMax: pixMax}
}

// newLogScale is newLinearScale on log10. Set sizes span three orders of
// magnitude and per-element costs span two, so a linear axis would compress
// everything interesting into the last tenth of the chart.
func newLogScale(dataMin, dataMax, pixMin, pixMax float64) scale {
	if dataMin <= 0 {
		dataMin = 1e-9
	}
	if dataMax <= dataMin {
		dataMax = dataMin * 10
	}
	return scale{dataMin: math.Log10(dataMin), dataMax: math.Log10(dataMax), pixMin: pixMin, pixMax: pixMax, logarithmic: true}
}

// at maps a data value to its device coordinate.
func (s scale) at(v float64) float64 {
	if s.logarithmic {
		if v <= 0 {
			v = 1e-9
		}
		v = math.Log10(v)
	}
	t := (v - s.dataMin) / (s.dataMax - s.dataMin)
	return s.pixMin + t*(s.pixMax-s.pixMin)
}

// niceTicks returns up to about count round values covering the scale's range.
func (s scale) niceTicks(count int) []float64 {
	if s.logarithmic {
		var out []float64
		for e := math.Floor(s.dataMin); e <= math.Ceil(s.dataMax); e++ {
			v := math.Pow(10, e)
			if math.Log10(v) >= s.dataMin-1e-9 && math.Log10(v) <= s.dataMax+1e-9 {
				out = append(out, v)
			}
		}
		return out
	}
	span := s.dataMax - s.dataMin
	step := niceStep(span / float64(count))
	var out []float64
	for v := math.Ceil(s.dataMin/step) * step; v <= s.dataMax+1e-9; v += step {
		out = append(out, v)
	}
	return out
}

// niceStep rounds a raw step up to 1, 2, 2.5 or 5 times a power of ten, which
// is what makes an axis readable rather than merely correct.
func niceStep(raw float64) float64 {
	if raw <= 0 {
		return 1
	}
	mag := math.Pow(10, math.Floor(math.Log10(raw)))
	switch n := raw / mag; {
	case n <= 1:
		return mag
	case n <= 2:
		return 2 * mag
	case n <= 2.5:
		return 2.5 * mag
	case n <= 5:
		return 5 * mag
	default:
		return 10 * mag
	}
}

// shortCount renders a set size as 1k, 16k, 262k, 2.1M.
func shortCount(n int) string {
	switch v := float64(n); {
	case v >= 1e6:
		return trimZero(strconv.FormatFloat(v/1e6, 'f', 1, 64)) + "M"
	case v >= 1e3:
		return trimZero(strconv.FormatFloat(v/1e3, 'f', 0, 64)) + "k"
	default:
		return strconv.Itoa(n)
	}
}

func trimZero(s string) string { return strings.TrimSuffix(s, ".0") }

// signedPercent renders a relative difference with an explicit sign, which
// matters here because the sign is the headline.
//
// A value that rounds to zero gets no sign at all. Without that, an axis tick
// computed as negative zero — which floating-point tick arithmetic produces
// routinely — comes out as the nonsense "+-0%".
func signedPercent(v float64, digits int) string {
	s := strconv.FormatFloat(v, 'f', digits, 64)
	if strings.Trim(s, "-0.") == "" {
		return strings.TrimPrefix(s, "-") + "%"
	}
	if v > 0 {
		return "+" + s + "%"
	}
	return s + "%"
}
