(() => {
  const state = {
    timers: []
  };

  const MIN_IDLE_MS = 5500;
  const MAX_IDLE_MS = 13000;
  const MIN_BREATH_MS = 1400;
  const MAX_BREATH_MS = 2600;
  const MIN_PARTICLE_GAP_MS = 24;
  const MAX_PARTICLE_GAP_MS = 55;
  const FIRE_COLORS = [
    ["#ff3b30", "#9d0f1c"],
    ["#ff4d3a", "#a90f1f"],
    ["#ff613d", "#b01323"],
    ["#ff7a40", "#c44916"]
  ];

  function randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function clearTimers() {
    state.timers.forEach((id) => window.clearTimeout(id));
    state.timers = [];
  }

  function setTimer(callback, delayMs) {
    const id = window.setTimeout(callback, delayMs);
    state.timers.push(id);
    return id;
  }

  function createFireParticle(mascot) {
    const fireLayer = mascot.querySelector("[data-scaley-fire-layer]");
    if (!fireLayer) {
      return;
    }

    const particle = document.createElement("span");
    particle.className = "scaley-fire-particle";
    const [particleColor, particleShadow] =
      FIRE_COLORS[randomBetween(0, FIRE_COLORS.length - 1)];
    particle.style.setProperty("--particle-size", `${randomBetween(3, 7)}px`);
    particle.style.setProperty("--particle-offset-y", `${randomBetween(-8, 8)}px`);
    particle.style.setProperty("--particle-travel-x", `${randomBetween(40, 88)}px`);
    particle.style.setProperty("--particle-drift-y", `${randomBetween(-6, 6)}px`);
    particle.style.setProperty("--particle-duration", `${randomBetween(380, 820)}ms`);
    particle.style.setProperty("--particle-color", particleColor);
    particle.style.setProperty("--particle-shadow", particleShadow);

    fireLayer.appendChild(particle);
    particle.addEventListener(
      "animationend",
      () => {
        particle.remove();
      },
      { once: true }
    );
  }

  function emitFirePacket(mascot) {
    const particlesThisTick = randomBetween(2, 4);
    for (let i = 0; i < particlesThisTick; i += 1) {
      createFireParticle(mascot);
    }
  }

  function scheduleBreath(mascot) {
    setTimer(() => {
      if (!document.body.contains(mascot)) {
        return;
      }

      mascot.classList.add("is-breathing");
      const burstEnd = performance.now() + randomBetween(MIN_BREATH_MS, MAX_BREATH_MS);

      const emit = () => {
        if (!document.body.contains(mascot)) {
          return;
        }

        emitFirePacket(mascot);

        if (performance.now() < burstEnd) {
          setTimer(emit, randomBetween(MIN_PARTICLE_GAP_MS, MAX_PARTICLE_GAP_MS));
          return;
        }

        mascot.classList.remove("is-breathing");
        scheduleBreath(mascot);
      };

      emit();
    }, randomBetween(MIN_IDLE_MS, MAX_IDLE_MS));
  }

  function initScaleyMascot() {
    clearTimers();

    const mascot = document.querySelector("[data-scaley-mascot]");
    if (!mascot) {
      return;
    }

    mascot.classList.remove("is-breathing");
    const fireLayer = mascot.querySelector("[data-scaley-fire-layer]");
    if (fireLayer) {
      fireLayer.replaceChildren();
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    scheduleBreath(mascot);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initScaleyMascot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initScaleyMascot, {
      once: true
    });
  } else {
    initScaleyMascot();
  }
})();
