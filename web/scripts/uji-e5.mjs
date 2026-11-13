// UJI COBA e5-small (tahap 2): apakah pasal yang BENAR ada di PERINGKAT 1?
// Jalankan dari folder web:  node scripts/uji-e5.mjs
// Catatan: teks potongan di bawah adalah parafrase umum untuk tes, BUKAN kutipan pasal resmi.
import os from "node:os";
import path from "node:path";
import { pipeline, env } from "@huggingface/transformers";

env.cacheDir = path.join(os.tmpdir(), "edupuan-model");

const t0 = Date.now();
const ekstrak = await pipeline("feature-extraction", "Xenova/multilingual-e5-small", { dtype: "q8" });
console.log(`Model dimuat dalam ${Date.now() - t0} ms\n`);

// Pertanyaan ke-i jawabannya potongan ke-i. Potongan 4 & 5 adalah pengecoh.
const pertanyaan = [
  "apakah penagih boleh menghubungi keluarga atau teman saya?",
  "berapa bunga maksimal pinjaman online per hari?",
  "apakah aplikasi pinjol boleh mengakses kontak di HP saya?",
  "bolehkah penagih datang ke rumah jam 11 malam?",
];
const potongan = [
  "Penagihan dilarang dilakukan kepada pihak selain konsumen, termasuk keluarga, kerabat, atau teman.",
  "Batas maksimum manfaat ekonomi untuk pendanaan konsumtif ditetapkan per hari dari nilai pendanaan.",
  "Penyelenggara hanya boleh mengakses kamera, mikrofon, dan lokasi pada perangkat pengguna.",
  "Penagihan dilarang dilakukan di luar pukul 08.00 sampai dengan pukul 20.00 waktu setempat.",
  "Resep rendang daging sapi khas Padang dengan santan dan cabai.",
  "Kereta api Makassar–Parepare berangkat setiap pagi dari stasiun Mandai.",
];

const embed = async (t) => (await ekstrak(t, { pooling: "mean", normalize: true })).tolist();
const mirip = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

// Embed SATU PER SATU (paling aman)
const vP = [];
for (const t of potongan) vP.push((await embed("passage: " + t))[0]);

let benar = 0;
for (let i = 0; i < pertanyaan.length; i++) {
  const vQ = (await embed("query: " + pertanyaan[i]))[0];
  const urut = vP
    .map((v, j) => ({ j, skor: mirip(vQ, v) }))
    .sort((a, b) => b.skor - a.skor);
  const ok = urut[0].j === i;
  if (ok) benar++;
  console.log(`${ok ? "✅" : "❌"} ${pertanyaan[i]}`);
  for (const u of urut.slice(0, 3)) {
    const tanda = u.j === i ? "← jawaban" : u.j >= 4 ? "(pengecoh)" : "";
    console.log(`     ${u.skor.toFixed(3)}  ${potongan[u.j].slice(0, 55)}… ${tanda}`);
  }
}
console.log(`\nPeringkat 1 benar: ${benar} dari ${pertanyaan.length}`);

// Cek teknis: embed SEKALIGUS (batch) harus sama dengan satu per satu
const vBatch = await embed(potongan.map((t) => "passage: " + t));
let selisih = 0;
for (let j = 0; j < potongan.length; j++)
  for (let k = 0; k < 384; k++) selisih = Math.max(selisih, Math.abs(vBatch[j][k] - vP[j][k]));
console.log(`Selisih batch vs satu-per-satu: ${selisih.toFixed(5)} (aman kalau < 0.01)`);