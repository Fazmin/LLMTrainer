import { describe, expect, it } from "vitest";
import {
  choicesGloss,
  fmtBytes,
  fmtCount,
  fmtDuration,
  fmtRoughTime,
  laneColor,
  laneSlot,
  natsToBits,
  readingGloss,
} from "../src/lib/format";

describe("format", () => {
  it("shortens big counts", () => {
    expect(fmtCount(950)).toBe("950");
    expect(fmtCount(12_345)).toBe("12.3k");
    expect(fmtCount(12_400_000)).toBe("12.4M");
    expect(fmtCount(150_000)).toBe("150k");
    expect(fmtCount(null)).toBe("–");
    expect(fmtCount(NaN)).toBe("–");
  });

  it("formats sizes the way people say them", () => {
    expect(fmtBytes(87_000_000)).toBe("87 MB");
    expect(fmtBytes(1_940_000_000)).toBe("1.9 GB");
    expect(fmtBytes(120_000)).toBe("120 KB");
    expect(fmtBytes(10)).toBe("1 KB");
    expect(fmtBytes(null)).toBe("–");
  });

  it("formats durations", () => {
    expect(fmtDuration(0)).toBe("0:00");
    expect(fmtDuration(65_000)).toBe("1:05");
    expect(fmtDuration(3_725_000)).toBe("1:02:05");
    expect(fmtDuration(null)).toBe("–");
  });

  it("describes time estimates roughly", () => {
    expect(fmtRoughTime(20)).toBe("under a minute");
    expect(fmtRoughTime(22 * 60)).toBe("about 22 minutes");
    expect(fmtRoughTime(3 * 3600)).toBe("about 3 hours");
    expect(fmtRoughTime(null)).toBe("–");
  });

  it("converts nats to bits and treats invalid numbers as missing", () => {
    expect(natsToBits(Math.LN2)).toBeCloseTo(1, 12);
    expect(natsToBits(null)).toBeNull();
    expect(natsToBits(NaN)).toBeNull();
  });

  it("explains a score in plain words", () => {
    expect(choicesGloss(5.58)).toMatch(/guessing at random/);
    expect(choicesGloss(Math.log(4.9))).toMatch(/about 4\.9 characters/);
    expect(choicesGloss(null)).toBe("");
  });

  it("gives human-scale reading amounts", () => {
    expect(readingGloss(0)).toBe("nothing yet");
    expect(readingGloss(12_400_000)).toBe("about 25 novels");
    expect(readingGloss(40_000)).toBe("about 20 pages");
  });

  it("keeps lane colours stable for the original lane names and folds past eight", () => {
    expect(laneSlot("stories", ["arithmetic", "stories"])).toBe(0);
    expect(laneSlot("arithmetic", ["stories", "arithmetic"])).toBe(1);
    // an unknown lane takes the first free slot, independent of listing order
    expect(laneSlot("notes", ["notes", "stories"])).toBe(laneSlot("notes", ["stories", "notes"]));
    expect(laneSlot("x", Array.from({ length: 12 }, (_, i) => `lane${i}`).concat("x"))).toBeGreaterThanOrEqual(0);
    expect(laneColor(8)).toBe("var(--muted)");
    expect(laneColor(0)).toBe("var(--lane-1)");
  });
});
