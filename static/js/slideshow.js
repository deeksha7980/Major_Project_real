document.addEventListener("DOMContentLoaded", () => {
  const slides = document.querySelectorAll(".slide");
  const dots   = document.querySelectorAll(".dot");
  let idx = 0;

  function show(n) {
    slides[idx].classList.remove("active");
    dots[idx].classList.remove("active");
    idx = (n + slides.length) % slides.length;
    slides[idx].classList.add("active");
    dots[idx].classList.add("active");
  }

  dots.forEach(d => d.addEventListener("click", () => show(+d.dataset.slide)));
  setInterval(() => show(idx + 1), 5000);
});