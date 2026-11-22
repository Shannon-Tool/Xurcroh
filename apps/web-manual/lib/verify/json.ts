/**
 * JSON reading and canonical writing that reproduce Python's `json` module byte for byte.
 *
 * Signatures cover `json.dumps(value, sort_keys=True, separators=(",", ":"))`. JavaScript's
 * own JSON functions cannot reproduce that: they spell numbers differently (`1e+22`
 * against `1e22`, `1.0` against `1`), lose integers above 2^53 and order keys by UTF-16
 * unit. So numbers are kept as text and respelled the way Python's `repr` would.
 */

export class JsonNumber {
  constructor(readonly text: string) {}
}

export type JsonObject = Map<string, JsonValue>;
export type JsonValue = null | boolean | string | JsonNumber | JsonValue[] | JsonObject;

export class JsonError extends Error {}

const WHITESPACE = new Set([" ", "\t", "\n", "\r"]);
const ESCAPES = new Map([
  ['"', '"'], ["\\", "\\"], ["/", "/"], ["b", "\b"], ["f", "\f"], ["n", "\n"], ["r", "\r"], ["t", "\t"],
]);
const NUMBER = /-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/y;

/** Spell a float the way Python's `repr(float)` does. */
function pythonFloat(token: string): string {
  const value = Number(token);
  if (!Number.isFinite(value)) throw new JsonError(`number ${token} is out of range`);
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";
  // toExponential() without an argument gives the shortest digits that round-trip,
  // which is also what Python prints.
  const [mantissa, exponentText] = Math.abs(value).toExponential().split("e") as [string, string];
  const digits = mantissa.replace(".", "");
  const exponent = Number(exponentText);
  const sign = value < 0 ? "-" : "";
  if (exponent < -4 || exponent >= 16) {
    const fraction = digits.length > 1 ? `.${digits.slice(1)}` : "";
    const magnitude = String(Math.abs(exponent)).padStart(2, "0");
    return `${sign}${digits[0]}${fraction}e${exponent < 0 ? "-" : "+"}${magnitude}`;
  }
  if (exponent < 0) return `${sign}0.${"0".repeat(-exponent - 1)}${digits}`;
  if (digits.length <= exponent + 1) {
    return `${sign}${digits}${"0".repeat(exponent + 1 - digits.length)}.0`;
  }
  return `${sign}${digits.slice(0, exponent + 1)}.${digits.slice(exponent + 1)}`;
}

function pythonNumber(token: string, isFloat: boolean): string {
  return isFloat ? pythonFloat(token) : BigInt(token).toString();
}

export function parseJson(text: string): JsonValue {
  let position = 0;

  const fail = (message: string): never => {
    throw new JsonError(`${message} at character ${position}`);
  };
  const skip = () => {
    while (position < text.length && WHITESPACE.has(text.charAt(position))) position += 1;
  };

  const readString = (): string => {
    position += 1;
    let result = "";
    for (;;) {
      if (position >= text.length) fail("unterminated string");
      const character = text.charAt(position);
      if (character === '"') {
        position += 1;
        return result;
      }
      if (character < " ") fail("control character in string");
      if (character !== "\\") {
        result += character;
        position += 1;
        continue;
      }
      const escape = text.charAt(position + 1);
      position += 2;
      if (escape === "u") {
        const hex = text.slice(position, position + 4);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("bad \\u escape");
        result += String.fromCharCode(parseInt(hex, 16));
        position += 4;
      } else {
        const simple = ESCAPES.get(escape);
        if (simple === undefined) fail("bad escape");
        result += simple;
      }
    }
  };

  const readValue = (): JsonValue => {
    skip();
    const character = text[position];
    if (character === "{") {
      position += 1;
      const object: JsonObject = new Map();
      skip();
      if (text[position] === "}") {
        position += 1;
        return object;
      }
      for (;;) {
        skip();
        if (text[position] !== '"') fail("expected a property name");
        const name = readString();
        // Python would silently keep the last of two equal names; a signed document
        // has no reason to contain them, so they are refused.
        if (object.has(name)) fail(`duplicate property ${JSON.stringify(name)}`);
        skip();
        if (text[position] !== ":") fail("expected ':'");
        position += 1;
        object.set(name, readValue());
        skip();
        if (text[position] === ",") {
          position += 1;
          continue;
        }
        if (text[position] === "}") {
          position += 1;
          return object;
        }
        fail("expected ',' or '}'");
      }
    }
    if (character === "[") {
      position += 1;
      const array: JsonValue[] = [];
      skip();
      if (text[position] === "]") {
        position += 1;
        return array;
      }
      for (;;) {
        array.push(readValue());
        skip();
        if (text[position] === ",") {
          position += 1;
          continue;
        }
        if (text[position] === "]") {
          position += 1;
          return array;
        }
        fail("expected ',' or ']'");
      }
    }
    if (character === '"') return readString();
    for (const [word, value] of [["true", true], ["false", false], ["null", null]] as const) {
      if (text.startsWith(word, position)) {
        position += word.length;
        return value;
      }
    }
    NUMBER.lastIndex = position;
    const match = NUMBER.exec(text);
    if (match === null) return fail("unexpected character");
    position += match[0].length;
    return new JsonNumber(pythonNumber(match[0], match[1] !== undefined || match[2] !== undefined));
  };

  const value = readValue();
  skip();
  if (position !== text.length) fail("unexpected text after the document");
  return value;
}

function quote(value: string, asciiOnly: boolean): string {
  let result = '"';
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    const code = value.charCodeAt(index);
    if (character === '"') result += '\\"';
    else if (character === "\\") result += "\\\\";
    else if (character === "\n") result += "\\n";
    else if (character === "\r") result += "\\r";
    else if (character === "\t") result += "\\t";
    else if (character === "\b") result += "\\b";
    else if (character === "\f") result += "\\f";
    else if (code < 0x20 || (asciiOnly && code > 0x7e)) {
      result += `\\u${code.toString(16).padStart(4, "0")}`;
    } else result += character;
  }
  return `${result}"`;
}

/** Order strings by Unicode code point, as Python does. */
function byCodePoint(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0) as number);
  const b = Array.from(right, (character) => character.codePointAt(0) as number);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    const difference = (a[index] as number) - (b[index] as number);
    if (difference !== 0) return difference;
  }
  return a.length - b.length;
}

/** `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=asciiOnly)`. */
export function canonicalText(value: JsonValue, asciiOnly: boolean): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return quote(value, asciiOnly);
  if (value instanceof JsonNumber) return value.text;
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalText(item, asciiOnly)).join(",")}]`;
  }
  const names = [...value.keys()].sort(byCodePoint);
  const members = names.map(
    (name) => `${quote(name, asciiOnly)}:${canonicalText(value.get(name) as JsonValue, asciiOnly)}`,
  );
  return `{${members.join(",")}}`;
}

export function canonicalBytes(value: JsonValue, asciiOnly: boolean): Uint8Array {
  return new TextEncoder().encode(canonicalText(value, asciiOnly));
}

export function isObject(value: JsonValue | undefined): value is JsonObject {
  return value instanceof Map;
}

/** A copy of `object` without the named members. */
export function without(object: JsonObject, ...names: string[]): JsonObject {
  return new Map([...object].filter(([name]) => !names.includes(name)));
}

/** Plain JavaScript data for display; numbers become strings so nothing is rounded. */
export function toDisplay(value: JsonValue): unknown {
  if (value instanceof JsonNumber) return value.text;
  if (Array.isArray(value)) return value.map(toDisplay);
  if (isObject(value)) {
    return Object.fromEntries([...value].map(([name, item]) => [name, toDisplay(item)]));
  }
  return value;
}
