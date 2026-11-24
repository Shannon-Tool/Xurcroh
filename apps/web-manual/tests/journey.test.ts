import { describe, expect, it } from "vitest";
import { executeShareGrid, freivaldsCheck, mod } from "@/lib/simulation/fortress";
import { SHOW_MODELS, SHOW_PRIME, getMatcher, getModel, openSession, runStep, seededRng, type TextModel } from "@/lib/simulation/show";
import { journeyOf, pictureJourneyOf } from "@/lib/show/journey";

const questions = SHOW_MODELS.filter((entry) => entry.kind === "text");

describe("reading the question on shares", () => {
  it("offers six prepared questions, each with its own full answer", () => {
    expect(questions).toHaveLength(6);
    const answers = questions.map((entry) => (getModel(entry.id) as TextModel).answer);
    expect(new Set(answers).size).toBe(6);
    for (const answer of answers) expect(answer.length).toBeGreaterThan(120);
  });

  it("turns each question into word counts and picks that question's own answer from the recombined scores", () => {
    const matcher = getMatcher();
    expect(matcher.questions).toEqual(questions.map((entry) => entry.id));
    expect(matcher.weight.rows).toBe(6);
    expect(matcher.weight.cols).toBe(matcher.vocabulary.length);
    const session = openSession(matcher, seededRng(4));
    questions.forEach((entry, index) => {
      const step = runStep(session, index);
      expect(step.rejected).toBe(false);
      expect(step.roles.every((role) => role.valid)).toBe(true);
      expect(matcher.pick(step.combined!)).toBe(entry.id);
      // The winner is clear, not a tie.
      const scores = Array.from(step.combined!).sort((a, b) => b - a);
      expect(scores[0]).toBeGreaterThan(scores[1]!);
      // A worker's piece of the word counts is not the word counts.
      expect(Array.from(step.activationShares[0])).not.toEqual(Array.from(matcher.input(index)));
    });
    expect(runStep(session, 2, 1).rejected).toBe(true);
  });
});

describe("the trip of one question", () => {
  it("carries the real pieces, products, checks, scores and the full answer", () => {
    for (const entry of questions) {
      const journey = journeyOf(entry.id);
      const model = getModel(entry.id) as TextModel;
      expect(journey.question).toBe(model.prompt);
      expect(journey.answer).toBe(model.answer);
      expect(journey.picked).toBe(entry.id);
      expect(journey.words.length).toBeGreaterThan(3);
      expect(journey.counts.reduce((sum, value) => sum + value, 0)).toBe(journey.words.length);
      for (let index = 0; index < journey.counts.length; index += 1) {
        expect((journey.pieces[0]![index]! + journey.pieces[1]![index]!) % SHOW_PRIME).toBe(journey.counts[index]);
      }
      expect(journey.products).toHaveLength(4);
      expect(journey.valid).toEqual([true, true, true, true]);
      for (let row = 0; row < 6; row += 1) {
        expect(journey.products.reduce((sum, product) => sum + product[row]!, 0) % SHOW_PRIME).toBe(journey.scores[row]);
      }
      expect(Math.max(...journey.scores)).toBe(journey.scores[questions.findIndex((item) => item.id === entry.id)]);
    }
  });

  it("is the same every time for one question and different between questions", () => {
    expect(journeyOf("text-shares")).toBe(journeyOf("text-shares"));
    expect(journeyOf("text-shares").pieces[0]).not.toEqual(journeyOf("text-tamper").pieces[0]);
    expect(journeyOf("nonsense").id).toBe("text-shares");
  });

  it("gives the labs small numbers taken from the question", () => {
    const journey = journeyOf("text-shares");
    // "How is my prompt…": H, o, w, i as positions in the alphabet, and as character codes.
    expect(journey.letters.slice(0, 4)).toEqual([8, 15, 23, 9]);
    expect(journey.bytes).toEqual([0x48, 0x6f, 0x77, 0x20]);
    expect(journey.codes).toHaveLength(4);
    for (const code of journey.codes) expect(code >= 0 && code < 16 && Number.isInteger(code)).toBe(true);
    expect(Number.isInteger(journey.hash)).toBe(true);
    expect(journeyOf("text-tamper").hash).not.toBe(journey.hash);
  });

  it("feeds the laboratory's big-number share grid and Freivalds check with the same question", () => {
    const journey = journeyOf("text-limits");
    const grid = executeShareGrid(journey.weightRows, journey.counts, 0);
    expect(grid.roles).toHaveLength(4);
    expect(grid.combined.map(Number)).toEqual(journey.scores);
    expect(() => executeShareGrid(journey.weightRows, journey.counts, 0, "1:0")).toThrow(/role 1:0/);
    const challenge = [11, 13, 17, 19, 23, 29];
    expect(freivaldsCheck(journey.weightRows, journey.counts, journey.scores, challenge)).toBe(true);
    const wrong = [...journey.scores];
    wrong[0]! += 1;
    expect(freivaldsCheck(journey.weightRows, journey.counts, wrong, challenge)).toBe(false);
    expect(mod(-1n) > 0n).toBe(true);
  });

  it("knows the two picture requests and their finished pictures", () => {
    const sunrise = pictureJourneyOf("picture-sunrise");
    expect(sunrise.request).toBe("A sunrise over the water");
    expect(sunrise.picture).toHaveLength(sunrise.width * sunrise.height * 3);
    expect(pictureJourneyOf("picture-city").request).toBe("A city at night under the moon");
    expect(pictureJourneyOf("nonsense").id).toBe("picture-sunrise");
  });
});
