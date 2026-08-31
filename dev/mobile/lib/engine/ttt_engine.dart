/// Tic-tac-toe model engine abstraction.
///
/// Two backends:
///  - [ExecuTorchEngine]: loads the `.pte` file via the native ExecuTorch
///    runtime (the production path on iOS/Android through dart:ffi).
///  - [DartReferenceEngine]: a pure-Dart reimplementation of the transformer
///    used on web/desktop where the native runtime isn't available.
library;

import 'dart:typed_data';

class TttForward {
  final double value; // sigmoid win-probability in [0,1]
  final Float64List policyLogits; // length 27, -inf on illegal cells
  TttForward(this.value, this.policyLogits);
}

abstract class TttEngine {
  Future<TttForward> forward(List<int> cells); // cells: 0/1/2
}

/// Placeholder for the native ExecuTorch-backed engine (see native/).
class ExecuTorchEngine implements TttEngine {
  @override
  Future<TttForward> forward(List<int> cells) async {
    throw UnimplementedError('native engine not wired in this build');
  }
}

/// 3D tic-tac-toe rules (lines, win detection).
class TttRules {
  static const n = 3;
  static final List<List<int>> _lines = _buildLines();

  static List<List<int>> _buildLines() {
    final lines = <List<int>>[];
    const dirs = [
      [1,0,0],[0,1,0],[0,0,1],
      [1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],
      [1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],
    ];
    for (var x = 0; x < 3; x++) {
      for (var y = 0; y < 3; y++) {
        for (var z = 0; z < 3; z++) {
          for (final d in dirs) {
            final pts = <int>[];
            for (var k = 0; k < 3; k++) {
              pts.add((x + d[0]*k) + 3*((y + d[1]*k) + 3*(z + d[2]*k)));
            }
            if (pts.every((p) => p >= 0 && p < 27)) lines.add(pts);
          }
        }
      }
    }
    return lines;
  }

  static List<List<int>> get lines => _lines;

  static int winner(List<int> cells) {
    for (final l in _lines) {
      final a = cells[l[0]];
      if (a != 0 && cells[l[1]] == a && cells[l[2]] == a) return a;
    }
    return 0;
  }
}
