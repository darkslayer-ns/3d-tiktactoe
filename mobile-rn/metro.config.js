const { getDefaultConfig } = require('expo/metro-config')

const config = getDefaultConfig(__dirname)

// `model.bin` never ships through Metro — the native module embeds it as a
// generated C header (scripts/embed_weights.py). Keep binary assets out of
// the JS bundle.
config.resolver.assetExts = config.resolver.assetExts.filter((e) => e !== 'bin')

module.exports = config