"use client";

import { journeyOf, pictureIds, pictureJourneyOf, questionIds } from "@/lib/show/journey";
import { pictureSetting, questionSetting, useJourney, usePictureJourney } from "@/lib/show/question";

/**
 * Chooses the prepared question (or picture request) that every lab and diagram follows.
 * The choice is shared: changing it here changes it everywhere on the site.
 */
export function QuestionPicker({ kind = "text" }: { kind?: "text" | "picture" }) {
  const question = useJourney();
  const picture = usePictureJourney();
  if (kind === "picture") {
    return (
      <label className="question-picker">Your picture request
        <select value={picture.id} onChange={(event) => pictureSetting.set(event.target.value)}>
          {pictureIds().map((id) => <option key={id} value={id}>{pictureJourneyOf(id).request}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="question-picker">Your question
      <select value={question.id} onChange={(event) => questionSetting.set(event.target.value)}>
        {questionIds().map((id) => <option key={id} value={id}>{journeyOf(id).question}</option>)}
      </select>
    </label>
  );
}
