const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const SAVE_PATH = path.join(__dirname, 'annotations.json');
let db = {};
if (fs.existsSync(SAVE_PATH)) {
  try { db = JSON.parse(fs.readFileSync(SAVE_PATH, 'utf8')); } catch(e) { db = {}; }
}

function save() {
  fs.writeFileSync(SAVE_PATH, JSON.stringify(db, null, 2));
}

// GET /api/annotations?user=Alice
app.get('/api/annotations', (req, res) => {
  const user = (req.query.user || '').trim();
  if (!user) return res.status(400).json({ error: 'user required' });
  res.json(db[user] || {});
});

// POST /api/annotations  { user, id, verdict, reasoning }
app.post('/api/annotations', (req, res) => {
  const { user, id, verdict, reasoning } = req.body;
  if (!user || id === undefined) return res.status(400).json({ error: 'user and id required' });
  if (!db[user]) db[user] = {};
  db[user][id] = { verdict, reasoning, updatedAt: new Date().toISOString() };
  save();
  res.json({ ok: true });
});

// GET /api/users  →  annotator progress summary
app.get('/api/users', (req, res) => {
  const summary = Object.entries(db).map(([user, anns]) => ({
    user,
    annotated: Object.values(anns).filter(a => a.verdict).length,
    unsafe: Object.values(anns).filter(a => a.verdict === 'Yes').length,
  }));
  res.json(summary);
});

// GET /api/export?user=Alice  or  /api/export  (all users)
app.get('/api/export', (req, res) => {
  const user = (req.query.user || '').trim();
  const users = user ? [user] : Object.keys(db);
  const rows = ['annotator,id,label,comments'];
  users.forEach(u => {
    Object.entries(db[u] || {}).forEach(([id, a]) => {
      const q = v => `"${String(v).replace(/"/g, '""')}"`;
      rows.push([q(u), id, a.verdict || '', q(a.reasoning || '')].join(','));
    });
  });
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="mhsafeeval_${user || 'all'}.csv"`);
  res.send(rows.join('\n'));
});

app.listen(PORT, () => {
  console.log(`\n✅  MHSafeEval → http://localhost:${PORT}`);
  console.log(`   Annotators so far: ${Object.keys(db).join(', ') || '(none yet)'}`);
  console.log(`   Progress summary:  http://localhost:${PORT}/api/users`);
  console.log(`   Export all data:   http://localhost:${PORT}/api/export\n`);
});