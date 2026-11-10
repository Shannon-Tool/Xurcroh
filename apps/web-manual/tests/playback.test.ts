import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { usePlayback } from "@/components/usePlayback";

describe("usePlayback", () => {
  it("steps and resets state", () => {
    const { result } = renderHook(() => usePlayback(3));
    act(() => result.current.onStep(1));
    expect(result.current.index).toBe(1);
    act(() => result.current.onReset());
    expect(result.current.index).toBe(0);
  });

  it("clamps the current state when the timeline shrinks", () => {
    const { result, rerender } = renderHook(({ maximum }) => usePlayback(maximum), {
      initialProps: { maximum: 8 },
    });
    act(() => result.current.onStep(8));
    expect(result.current.index).toBe(8);
    rerender({ maximum: 2 });
    expect(result.current.index).toBe(2);
  });
});
