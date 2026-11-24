import { ScrambleText } from "@/components/ScrambleText";

/** One form the data takes on its way: readable, numbers, shuffled, pieces, sealed, and back. */
export interface FlowForm {
  id: string;
  label: string;
  value: string;
  /** readable (good), random pieces (accent), sealed (cipher), failed (warn) or plain numbers. */
  tone: "good" | "accent" | "cipher" | "warn" | "plain";
}

/**
 * The same data in every form it takes, left to right, with the current form lit. This is where the
 * reader sees readable text become numbers, pieces and sealed bytes, and come back as an answer.
 */
export function FlowRibbon({ forms, current, label = "The data in each form it takes" }: { forms: FlowForm[]; current: string[]; label?: string }) {
  return (
    <ol className="ribbon" aria-label={label}>
      {forms.map((form) => {
        const on = current.includes(form.id);
        return (
          <li key={form.id} className={`ribbon-form ribbon-${form.tone}${on ? " on" : ""}`} aria-current={on ? "step" : undefined}>
            <span className="ribbon-label">{form.label}</span>
            <ScrambleText className="ribbon-value" text={form.value} />
          </li>
        );
      })}
    </ol>
  );
}
