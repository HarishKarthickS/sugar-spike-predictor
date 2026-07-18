/**
 * Client-side validation for the prediction form.
 * Server-side checks in app/validation.py remain the source of truth.
 */
(function () {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const rules = {
    age: { min: 1, max: 120, label: "Age" },
    bmi: { min: 10, max: 60, label: "BMI" },
    a1c: { min: 3.5, max: 15, label: "A1c (%)" },
    fasting_glucose: { min: 50, max: 400, label: "Fasting glucose" },
    insulin_level: { min: 0, max: 300, label: "Insulin level" },
    heart_rate: { min: 40, max: 200, label: "Heart rate" },
    current_glucose: { min: 40, max: 600, label: "Current glucose" },
    glucose_trend: { min: -30, max: 30, label: "Glucose trend" },
    calories: { min: 0, max: 3000, label: "Calories" },
    carbs: { min: 0, max: 500, label: "Carbohydrates" },
    protein: { min: 0, max: 300, label: "Protein" },
    fat: { min: 0, max: 300, label: "Fat" },
    fiber: { min: 0, max: 100, label: "Fiber" },
  };

  function feedbackEl(input) {
    const id = input.getAttribute("aria-describedby");
    if (!id) return null;
    return document.getElementById(id.split(" ")[0]);
  }

  function setInvalid(input, message) {
    input.classList.add("is-invalid");
    input.classList.remove("is-valid");
    input.setAttribute("aria-invalid", "true");
    const tip = feedbackEl(input);
    if (tip) tip.textContent = message;
  }

  function setValid(input) {
    input.classList.remove("is-invalid");
    input.classList.add("is-valid");
    input.setAttribute("aria-invalid", "false");
    const tip = feedbackEl(input);
    if (tip && tip.dataset.default) tip.textContent = tip.dataset.default;
  }

  function clearState(input) {
    input.classList.remove("is-invalid", "is-valid");
    input.removeAttribute("aria-invalid");
  }

  function validateNumber(input, rule) {
    const raw = input.value.trim();
    if (!raw) {
      setInvalid(input, `${rule.label} is required.`);
      return false;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      setInvalid(input, `${rule.label} must be a number.`);
      return false;
    }
    if (value < rule.min || value > rule.max) {
      setInvalid(input, `${rule.label} must be between ${rule.min} and ${rule.max}.`);
      return false;
    }
    setValid(input);
    return true;
  }

  function validateSelect(input, allowed, label) {
    if (!allowed.includes(input.value)) {
      setInvalid(input, `Select a valid ${label}.`);
      return false;
    }
    setValid(input);
    return true;
  }

  function crossChecks() {
    let ok = true;
    const carbs = form.elements.namedItem("carbs");
    const fiber = form.elements.namedItem("fiber");
    const calories = form.elements.namedItem("calories");
    const protein = form.elements.namedItem("protein");
    const fat = form.elements.namedItem("fat");

    const carbsVal = Number(carbs.value);
    const fiberVal = Number(fiber.value);
    if (Number.isFinite(carbsVal) && Number.isFinite(fiberVal) && fiberVal > carbsVal) {
      setInvalid(fiber, "Fiber cannot be greater than carbohydrates.");
      ok = false;
    }

    const cal = Number(calories.value);
    const p = Number(protein.value);
    const f = Number(fat.value);
    if ([cal, carbsVal, p, f].every(Number.isFinite)) {
      const estimated = 4 * carbsVal + 4 * p + 9 * f;
      if (cal > 0 && estimated > cal * 1.6 + 50) {
        setInvalid(calories, "Calories look too low for the entered macros.");
        ok = false;
      } else if (cal >= 50 && estimated < cal * 0.35) {
        setInvalid(calories, "Calories look high for the entered macros.");
        ok = false;
      }
    }
    return ok;
  }

  function validateAll() {
    let ok = true;
    Object.keys(rules).forEach((name) => {
      const input = form.elements.namedItem(name);
      if (!input) return;
      if (!validateNumber(input, rules[name])) ok = false;
    });
    const gender = form.elements.namedItem("gender");
    const mealType = form.elements.namedItem("meal_type");
    if (!validateSelect(gender, ["Male", "Female", "Other"], "gender")) ok = false;
    if (!validateSelect(mealType, ["Breakfast", "Lunch", "Dinner", "Snack"], "meal type")) ok = false;
    if (!crossChecks()) ok = false;
    return ok;
  }

  form.setAttribute("novalidate", "novalidate");

  form.addEventListener("submit", (event) => {
    if (!validateAll()) {
      event.preventDefault();
      const firstInvalid = form.querySelector(".is-invalid");
      if (firstInvalid) firstInvalid.focus();
      const banner = document.getElementById("client-form-error");
      if (banner) {
        banner.classList.remove("d-none");
        banner.textContent = "Please fix the highlighted fields and try again.";
      }
    }
  });

  form.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
    if (target.name in rules) validateNumber(target, rules[target.name]);
    if (target.name === "gender") validateSelect(target, ["Male", "Female", "Other"], "gender");
    if (target.name === "meal_type") {
      validateSelect(target, ["Breakfast", "Lunch", "Dinner", "Snack"], "meal type");
    }
    if (["carbs", "fiber", "calories", "protein", "fat"].includes(target.name)) {
      crossChecks();
    }
  });

  form.addEventListener("reset", () => {
    Array.from(form.elements).forEach((el) => {
      if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) clearState(el);
    });
  });
})();
