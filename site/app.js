const presets = {
  summary: {
    query: "把这封客户邮件概括成三点",
    values: [74, 13, 7, 4, 2],
    selected: 0,
    model: "tier_1 → fast-model"
  },
  contract: {
    query: "审查这份采购合同的违约风险",
    values: [3, 6, 9, 14, 68],
    selected: 4,
    model: "tier_5 → strong-model"
  },
  extract: {
    query: "提取发票字段并输出标准 JSON",
    values: [9, 14, 63, 10, 4],
    selected: 2,
    model: "tier_3 → balanced-model"
  }
};

const probabilityList = document.querySelector("#probabilities");
function renderPreset(name) {
  const preset = presets[name];
  document.querySelector("#demo-query").textContent = preset.query;
  document.querySelector("#demo-model").textContent = preset.model;
  document.querySelector("#demo-confidence").textContent = `${preset.values[preset.selected]}% CONF.`;
  probabilityList.innerHTML = preset.values.map((value, index) => `
    <div class="prob-row ${index === preset.selected ? "selected" : ""}">
      <span>tier_${index + 1}</span>
      <span class="prob-track"><i style="--value:${value}%"></i></span>
      <span>${value}%</span>
    </div>`).join("");
  document.querySelectorAll("[data-preset]").forEach(button => {
    button.classList.toggle("active", button.dataset.preset === name);
  });
}
document.querySelectorAll("[data-preset]").forEach(button => {
  button.addEventListener("click", () => renderPreset(button.dataset.preset));
});
renderPreset("contract");

const topbar = document.querySelector("#topbar");
addEventListener("scroll", () => topbar.classList.toggle("scrolled", scrollY > 16), { passive: true });

const menuButton = document.querySelector("#menu-button");
const navLinks = document.querySelector("#nav-links");
menuButton.addEventListener("click", () => {
  const open = navLinks.classList.toggle("open");
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.textContent = open ? "×" : "☰";
});
navLinks.querySelectorAll("a").forEach(link => link.addEventListener("click", () => {
  navLinks.classList.remove("open");
  menuButton.setAttribute("aria-expanded", "false");
  menuButton.textContent = "☰";
}));

const installText = `git clone https://github.com/HuggingAI/RouteTune.git\ncd RouteTune\ncp .env.example .env\ndocker compose up --build -d`;
document.querySelector("#copy-button").addEventListener("click", async event => {
  try {
    await navigator.clipboard.writeText(installText);
    event.currentTarget.textContent = "已复制 ✓";
    setTimeout(() => { event.currentTarget.textContent = "复制命令"; }, 1600);
  } catch (_) {
    event.currentTarget.textContent = "请手动复制";
  }
});

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
if (reduceMotion || !("IntersectionObserver" in window)) {
  document.querySelectorAll(".reveal").forEach(item => item.classList.add("visible"));
} else {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: .12 });
  document.querySelectorAll(".reveal").forEach(item => observer.observe(item));
}
