// Polyfill globalThis.File for Node 18.15 before any ESM module loads.
// undici v6+ references File at import time; Node only exposes it globally from 18.17+.
const { File } = require('buffer');
if (typeof globalThis.File === 'undefined') globalThis.File = File;
