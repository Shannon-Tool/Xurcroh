import { useSyncExternalStore } from "react";
import { createSetting } from "@/lib/scene/settings";
import { journeyOf, pictureIds, pictureJourneyOf, questionIds, type Journey, type PictureJourney } from "@/lib/show/journey";

/** The prepared question every lab and diagram on the site is currently following, remembered per browser. */
export const questionSetting = createSetting<string>("xurcroh-question", "text-shares", questionIds());
/** The prepared picture request the picture lab is following. */
export const pictureSetting = createSetting<string>("xurcroh-picture", "picture-sunrise", pictureIds());

export function useJourney(): Journey {
  const id = useSyncExternalStore(questionSetting.subscribe, questionSetting.get, () => "text-shares");
  return journeyOf(id);
}

export function usePictureJourney(): PictureJourney {
  const id = useSyncExternalStore(pictureSetting.subscribe, pictureSetting.get, () => "picture-sunrise");
  return pictureJourneyOf(id);
}
