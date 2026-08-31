import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'engine/ttt_engine.dart';

void main() {
  runApp(const TttApp());
}

class TttApp extends StatelessWidget {
  const TttApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '3D Tic-Tac-Toe',
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        scaffoldBackgroundColor: const Color(0xFF0a0e1a),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF22d3ee),
          secondary: Color(0xFFf472b6),
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late List<int> _cells;
  int _turn = 1; // 1 = X (human), 2 = O (AI)
  int _status = 0; // 0 playing, 1/2 winner, 3 draw
  bool _aiThinking = false;
  TttEngine? _engine;

  @override
  void initState() {
    super.initState();
    _cells = List.filled(27, 0);
    _initEngine();
  }

  Future<void> _initEngine() async {
    // Reference engine: pure Dart forward pass. On iOS/Android this is
    // swapped for the ExecuTorch runtime (see native/README) via FFI.
    _engine = DartReferenceEngine();
    setState(() {});
  }

  void _reset() {
    setState(() {
      _cells = List.filled(27, 0);
      _turn = 1;
      _status = 0;
    });
  }

  Future<void> _play(int idx) async {
    if (_status != 0 || _aiThinking || _cells[idx] != 0 || _turn != 1) return;
    setState(() => _cells[idx] = 1);
    final w = TttRules.winner(_cells);
    if (w != 0) {
      setState(() => _status = w);
      return;
    }
    if (!_cells.contains(0)) {
      setState(() => _status = 3);
      return;
    }
    _turn = 2;
    await _aiMove();
  }

  Future<void> _aiMove() async {
    setState(() => _aiThinking = true);
    final idx = await _searchMove(_cells, 2, 3, 3);
    if (mounted) {
      setState(() {
        if (idx >= 0 && _cells[idx] == 0) _cells[idx] = 2;
        _aiThinking = false;
        _turn = 1;
        final w = TttRules.winner(_cells);
        if (w != 0) {
          _status = w;
        } else if (!_cells.contains(0)) {
          _status = 3;
        }
      });
    }
  }

  /// Expectimax: AI plays the move maximizing expected value, opponent's
  /// replies weighted by the model's own policy (like the backend lookahead).
  Future<int> _searchMove(
      List<int> cells, int ai, int depth, int topK) async {
    final engine = _engine;
    if (engine == null) return -1;
    var best = -1;
    var bestV = double.negativeInfinity;
    for (var m = 0; m < 27; m++) {
      if (cells[m] != 0) continue;
      cells[m] = ai;
      final v = await _eval(cells, ai, depth - 1, topK);
      cells[m] = 0;
      if (v > bestV) {
        bestV = v;
        best = m;
      }
    }
    return best;
  }

  Future<double> _eval(
      List<int> cells, int ai, int depth, int topK) async {
    final w = TttRules.winner(cells);
    if (w != 0) return w == ai ? 1.0 : 0.0;
    if (!cells.contains(0)) return 0.5;
    final opp = ai == 1 ? 2 : 1;
    // opponent to move: predict their move via the model's policy
    final f = await _engine!.forward(cells);
    // softmax over legal cells -> candidate replies
    final probs = <int, double>{};
    var sum = 0.0;
    final legal = <int>[];
    for (var i = 0; i < 27; i++) {
      if (cells[i] == 0) legal.add(i);
    }
    final logits = f.policyLogits;
    for (final i in legal) {
      final e = math.exp(logits[i]);
      probs[i] = e;
      sum += e;
    }
    final replies = legal.map((i) => (i, probs[i]! / sum)).toList()
      ..sort((a, b) => b.$2.compareTo(a.$2));
    final top = replies.take(topK).toList();
    final wsum = top.fold(0.0, (s, r) => s + r.$2);
    var exp = 0.0;
    for (final (i, pr) in top) {
      cells[i] = opp;
      final sub = depth > 0 ? await _searchMove(cells, ai, depth, topK) : -1;
      double rv;
      if (sub >= 0) {
        cells[sub] = ai;
        rv = await _eval(cells, ai, depth - 1, topK);
        cells[sub] = 0;
      } else {
        rv = await _engine!.forward(cells).then((ff) => ff.value);
      }
      cells[i] = 0;
      exp += (pr / wsum) * rv;
    }
    return exp;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            children: [
              const Text('3D Tic-Tac-Toe',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text(_statusText(),
                  style: const TextStyle(color: Color(0xFF94a3b8))),
              const SizedBox(height: 8),
              Expanded(child: CubeBoard(cells: _cells, onTap: _play)),
              const SizedBox(height: 8),
              FilledButton(onPressed: _reset, child: const Text('New game')),
            ],
          ),
        ),
      ),
    );
  }

  String _statusText() {
    if (_aiThinking) return 'AI is thinking…';
    switch (_status) {
      case 1:
        return 'You win!';
      case 2:
        return 'AI wins';
      case 3:
        return 'Draw';
      default:
        return _turn == 1 ? 'Your turn (X)' : 'AI turn';
    }
  }
}

/// A simple pseudo-3D isometric render of the 3x3x3 cube.
class CubeBoard extends StatelessWidget {
  final List<int> cells;
  final ValueChanged<int> onTap;
  const CubeBoard({super.key, required this.cells, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapUp: (d) {
        // crude projection: nearest empty cell around the tap
        final pos = d.localPosition;
        final center = 150.0;
        final dx = (pos.dx - center) / 70;
        final dy = (pos.dy - center) / 70;
        final x = (dx + 1).round().clamp(0, 2);
        final y = (dy + 1).round().clamp(0, 2);
        // z picked by scanning (simplified)
        for (var z = 0; z < 3; z++) {
          final idx = x + 3 * (y + 3 * z);
          if (cells[idx] == 0) {
            onTap(idx);
            return;
          }
        }
        onTap(x + 3 * (y + 3 * 1));
      },
      child: CustomPaint(
        size: const Size(300, 300),
        painter: _CubePainter(cells),
      ),
    );
  }
}

class _CubePainter extends CustomPainter {
  final List<int> cells;
  _CubePainter(this.cells);

  Offset _proj(int x, int y, int z, double cx, double cy) {
    final isoX = (x - y) * 50.0 + cx;
    final isoY = (x + y) * 25.0 - z * 55.0 + cy;
    return Offset(isoX, isoY);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    for (var z = 0; z < 3; z++) {
      for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 3; x++) {
          final c = _proj(x, y, z, cx, cy);
          final val = cells[x + 3 * (y + 3 * z)];
          final paint = Paint()
            ..color = val == 1
                ? const Color(0xFF22d3ee)
                : val == 2
                    ? const Color(0xFFf472b6)
                    : const Color(0xFF1e293b)
            ..style = PaintingStyle.fill;
          canvas.drawRRect(
              RRect.fromRectAndRadius(
                  Rect.fromCenter(center: c, width: 34, height: 34),
                  const Radius.circular(6)),
              paint);
          if (val != 0) {
            final tp = TextPainter(
              text: TextSpan(
                text: val == 1 ? 'X' : 'O',
                style: TextStyle(
                    color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
              ),
              textDirection: TextDirection.ltr,
            )..layout();
            tp.paint(canvas, c - Offset(tp.width / 2, tp.height / 2));
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant _CubePainter old) => old.cells != cells;
}