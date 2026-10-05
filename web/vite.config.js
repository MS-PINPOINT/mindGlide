import { defineConfig } from "vite";
import { readFileSync } from "node:fs";

const config = JSON.parse(
  readFileSync(new URL("./vercel.json", import.meta.url)),
);
const isolation = Object.fromEntries(
  config.headers[0].headers.map((item) => [item.key, item.value]),
);
const workerPolicy = config.headers.find((item) => item.source === "/worker.js")
  .headers[0].value;
function workerHeaders(directory) {
  return (req, res, next) => {
    if (req.url?.split("?")[0] !== "/worker.js") return next();
    for (const [key, value] of Object.entries(isolation))
      res.setHeader(key, value);
    res.setHeader("Content-Security-Policy", workerPolicy);
    res.setHeader("Content-Type", "application/javascript");
    res.end(readFileSync(new URL(`./${directory}/worker.js`, import.meta.url)));
  };
}

export default defineConfig({
  plugins: [
    {
      name: "worker-csp",
      configureServer(server) {
        server.middlewares.use(workerHeaders("public"));
      },
      configurePreviewServer(server) {
        server.middlewares.use(workerHeaders("dist"));
      },
    },
  ],
  server: { headers: isolation },
  preview: { headers: isolation },
  build: { sourcemap: false, target: "es2022" },
});
