/** Motion constants: one orchestrated moment (current flowing along an incident's decision line) and
 * quiet transitions for state changes. Reduced motion turns both off. */
export const DURATION = { reveal: 0.35, flow: 1.1, step: 0.12 } as const;
export const EASE = [0.2, 0.7, 0.2, 1] as const;
