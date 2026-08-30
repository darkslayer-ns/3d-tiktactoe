/**
 * Jest config: game-logic + AI modules are pure TypeScript (no RN/three
 * imports), so they run in node with ts-jest. UI/native modules are excluded.
 */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  transform: {
    '^.+\\.ts$': ['ts-jest', { tsconfig: { module: 'commonjs' } }],
  },
}