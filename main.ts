// GTA VI trophy rarity forecaster (Node 22, standard library only).
//
// For every trophy, forecast five quantiles of its Exophase earn rate at the
// case's target, as a percent. The score is the mean pinball loss on the logit
// scale, so the quantiles must be the ones the forecaster really believes.
//
// Model, per trophy, fitted on the case's whole history:
//   - Work on y = logit(rate), against x = ln(1 + hours since launch
//     2026-11-19T00:00:00Z). Log-time is the right clock for a launch: rates
//     move fast in the first hours and then flatten as the player base
//     matures, so a line on log-time already bends the future down.
//   - Weighted least squares, weights decaying with a 24h half-life, so the
//     recent reads (the player base the target inherits) dominate while the
//     launch-night reads still anchor the level.
//   - The slope is shrunk toward its tier's mean slope (platinum, gold,
//     silver and bronze cohorts move very differently, and a single noisy
//     trophy must not extrapolate a wild slope). The extrapolation is then
//     damped by 0.75: growth decelerates faster than log-time once the
//     launch cohort has played through the content.
//   - The median is clamped to sane bounds: never below 70% of the last
//     read (rates fall only when many new players join at once) and never
//     more than halfway to 100% above the last read in one horizon.
//   - The spread combines the fit's residual (how wobbly this trophy's
//     history is) with a random-walk term, 0.5 logit units per square-root
//     day, and a floor, so the 90% band is honest about a multi-day horizon
//     instead of razor-thin like the flat baseline's practice band.
//
// With fewer than 3 usable reads, or a flat history (every read the same,
// as on launch night before the refresh), the fit has no slope and the
// forecast falls back to the last rate with the random-walk spread — the
// flat baseline, slightly wider.
//
// Everything is a pure function of the case file: deterministic, no
// network, no clock, no randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Launch moment: the clock every read is measured against. */
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** Recency half-life of a read's weight, in hours. */
const HALF_LIFE_H = 12;
/** Only reads this recent (hours before the last read) inform the slope:
 * the launch's early phase was far steeper than its tail, and a slope fitted
 * over the whole history overshoots the future. */
const SLOPE_WINDOW_H = 48;
/** Prior strength, in read-equivalents, pulling a slope to its tier mean. */
const TIER_PRIOR = 4;
/** Damping on the extrapolated growth: the future flattens faster than log-time. */
const DAMP = 0.75;
/** Random-walk spread, in logit units per square-root day. */
const SIGMA = 0.5;
/** Floor on the spread, in logit units: no trophy is ever that certain. */
const SPREAD_FLOOR = 0.15;
/** Floor on a trophy's median, as a multiple of its last rate. */
const MEDIAN_FLOOR = 0.7;
/** Cap on a trophy's median gain, as a share of the headroom to 100%. */
const MEDIAN_HEADROOM = 0.35;

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Array<{ id: number; tier: string }>;
  history: Array<{ at: string; rates: Array<number | null> }>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const hoursSinceLaunch = (iso: string): number =>
  (Date.parse(iso) - LAUNCH_MS) / 3600000;

interface Point {
  x: number;
  y: number;
  w: number;
}

/**
 * Weighted least squares of y on x. Returns the slope, the intercept, the
 * weighted residual variance, and the effective sample size.
 */
function wls(points: Point[]): { slope: number; intercept: number; residVar: number; nEff: number } {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  const xBar = sx / sw;
  const yBar = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.x - xBar) * (p.x - xBar);
    sxy += p.w * (p.x - xBar) * (p.y - yBar);
  }
  const slope = sxx > 1e-9 ? sxy / sxx : 0;
  const intercept = yBar - slope * xBar;
  let sse = 0;
  for (const p of points) {
    const r = p.y - (intercept + slope * p.x);
    sse += p.w * r * r;
  }
  const nEff = (sw * sw) / points.reduce((a, p) => a + p.w * p.w, 0);
  const residVar = points.length > 2 ? sse / sw : 0;
  return { slope, intercept, residVar, nEff };
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error('case has no history');
  const lastRead = c.history[c.history.length - 1];
  const lastT = hoursSinceLaunch(lastRead.at);
  const targetT = hoursSinceLaunch(c.target);
  const dDays = Math.max(0, c.horizon_minutes / 1440);
  const xTarget = Math.log(1 + Math.max(0, targetT));

  // Per-trophy fits, then tier mean slopes for shrinkage.
  const fits = c.trophies.map((t, i) => {
    const points: Point[] = [];
    for (const read of c.history) {
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      const tH = hoursSinceLaunch(read.at);
      const ageH = Math.max(0, lastT - tH);
      if (ageH > SLOPE_WINDOW_H) continue;
      points.push({
        x: Math.log(1 + Math.max(0, tH)),
        y: logit(r),
        w: Math.pow(0.5, ageH / HALF_LIFE_H),
      });
    }
    const lastRate = lastRead.rates[i] ?? 0;
    if (points.length < 3) return { tier: t.tier, slope: 0, intercept: logit(lastRate), residVar: 0, nEff: 0, lastRate };
    const f = wls(points);
    return { tier: t.tier, slope: f.slope, intercept: f.intercept, residVar: f.residVar, nEff: f.nEff, lastRate };
  });

  // Tier mean slope, weighted by each trophy's effective sample size.
  const tierSlope = new Map<string, number>();
  for (const tier of new Set(fits.map((f) => f.tier))) {
    let sw = 0;
    let ss = 0;
    for (const f of fits) {
      if (f.tier !== tier) continue;
      const w = Math.max(0, f.nEff);
      sw += w;
      ss += w * f.slope;
    }
    tierSlope.set(tier, sw > 0 ? ss / sw : 0);
  }

  return c.trophies.map((t, i) => {
    const f = fits[i];
    const lastLogit = logit(f.lastRate);
    // Shrink the slope toward the tier mean; a trophy with little data
    // borrows its tier's behaviour, one with a long stable history keeps
    // its own.
    const prior = tierSlope.get(f.tier) ?? 0;
    const slope = (f.nEff * f.slope + TIER_PRIOR * prior) / (f.nEff + TIER_PRIOR);
    // Median: start from the LAST observed rate — the one hard fact — and
    // move it by the (shrunk, damped) slope over the horizon. Never anchor
    // on the fitted line's value: on a concave curve the full-history line
    // sits above the last read and extrapolating from it overshoots.
    const xLast = Math.log(1 + Math.max(0, lastT));
    let medianLogit = lastLogit + DAMP * slope * (xTarget - xLast);
    // Sanity bounds: a rate cannot fall far in a few days, nor eat more
    // than half its headroom to 100% in one horizon.
    const floorLogit = logit(f.lastRate * MEDIAN_FLOOR);
    const capLogit = logit(f.lastRate + (100 - f.lastRate) * MEDIAN_HEADROOM);
    medianLogit = Math.min(capLogit, Math.max(floorLogit, medianLogit));
    // Spread: fit residual plus a random walk widening with the horizon.
    const spread = Math.sqrt(
      Math.max(0, f.residVar) + SIGMA * SIGMA * dDays + SPREAD_FLOOR * SPREAD_FLOOR,
    );
    const median = sigmoidPct(medianLogit);
    const q = Z.map((z) => round4(clampPct(sigmoidPct(medianLogit + z * spread))));
    // Keep the row non-decreasing and the median where it belongs: the
    // middle quantile is the median itself.
    q[2] = round4(clampPct(median));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
