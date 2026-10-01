import { describe, expect, it } from "vitest";
import { configDiff, flatten, formatValue, keyLabel } from "../src/lib/diff";

describe("config diff", () => {
  it("flattens nested objects into dotted keys", () => {
    expect(flatten({ a: 1, b: { c: 2, d: { e: 3 } } })).toEqual({ a: 1, "b.c": 2, "b.d.e": 3 });
  });

  it("returns only the keys that differ", () => {
    const a = { train: { lr: 0.001, chunk: 512, growth: { k: 1 } }, model: { dModel: 256 } };
    const b = { train: { lr: 0.002, chunk: 512, growth: { k: 1 } }, model: { dModel: 256 } };
    const c = { train: { lr: 0.002, chunk: 1024, growth: { k: 1 } }, model: { dModel: 256 } };
    expect(configDiff([a, b])).toEqual([{ key: "train.lr", values: [0.001, 0.002] }]);
    expect(configDiff([a, b, c]).map((r) => r.key)).toEqual(["train.chunk", "train.lr"]);
    expect(configDiff([a, a])).toEqual([]);
  });

  it("treats a key missing from one config as different", () => {
    expect(configDiff([{ x: 1 }, { x: 1, y: 2 }])).toEqual([{ key: "y", values: [undefined, 2] }]);
  });

  it("gives readable labels and values", () => {
    expect(keyLabel("train.lr")).toBe("Learning rate");
    expect(keyLabel("train.growth.birthGate")).toBe("Birth gate");
    expect(formatValue(0.0123456)).toBe("0.01235");
    expect(formatValue(2048)).toBe("2,048");
    expect(formatValue(true)).toBe("yes");
    expect(formatValue("dense_masked")).toBe("dense masked");
    expect(formatValue(undefined)).toBe("–");
  });
});
