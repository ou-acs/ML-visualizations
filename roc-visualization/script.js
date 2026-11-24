// Interactive theoretical ROC curve with two controls:
// 1. AUROC controls how well-separated the two score distributions are.
// 2. Threshold moves the operating point along the selected ROC curve.
//
// Model assumption:
//   Score | negative class ~ Normal(-d/2, 1)
//   Score | positive class ~ Normal(+d/2, 1)
//
// For equal-variance normal distributions:
//   AUC = Phi(d / sqrt(2))
// Therefore:
//   d = sqrt(2) * Phi_inverse(AUC)
//
// At threshold c:
//   FPR = P(score >= c | negative) = 1 - Phi(c + d/2)
//   TPR = P(score >= c | positive) = 1 - Phi(c - d/2)
//
// The slider's threshold value is mapped to a score cutoff from +4 to -4.
// This keeps the displayed threshold control between 0 and 1 while covering
// nearly the full ROC curve.

function normalCDF(x) {
  // Abramowitz-Stegun approximation to the standard normal CDF.
  const absX = Math.abs(x);
  const t = 1 / (1 + 0.2316419 * absX);
  const d = 0.3989422804014327 * Math.exp(-absX * absX / 2);
  const p = 1 - d * t * (
    0.319381530 +
    t * (-0.356563782 +
    t * (1.781477937 +
    t * (-1.821255978 +
    t * 1.330274429)))
  );
  return x >= 0 ? p : 1 - p;
}

function inverseNormalCDF(p) {
  // Acklam's rational approximation to the inverse standard normal CDF.
  if (p <= 0 || p >= 1) {
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    throw new Error("Probability must be between 0 and 1.");
  }

  const a = [
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00
  ];
  const b = [
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01
  ];
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00
  ];
  const d = [
     7.784695709041462e-03,
     3.224671290700398e-01,
     2.445134137142996e+00,
     3.754408661907416e+00
  ];

  const low = 0.02425;
  const high = 1 - low;
  let q, r;

  if (p < low) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }

  if (p > high) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }

  q = p - 0.5;
  r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

function parametersFromAUC(auc) {
  // Separation between the class-conditional normal distributions.
  const separation = Math.SQRT2 * inverseNormalCDF(auc);
  return {
    separation,
    negativeMean: -separation / 2,
    positiveMean: separation / 2
  };
}

function calculateRates(thresholdControl, params) {
  // Map UI threshold [0,1] to a score cutoff [+4,-4].
  const cutoff = 4 - 8 * thresholdControl;

  const fpr = 1 - normalCDF(cutoff - params.negativeMean);
  const tpr = 1 - normalCDF(cutoff - params.positiveMean);

  return {
    fpr: Math.max(0, Math.min(1, fpr)),
    tpr: Math.max(0, Math.min(1, tpr)),
    cutoff
  };
}

function buildCurve(params) {
  const fpr = [];
  const tpr = [];
  const pointCount = 500;

  // Sweep score cutoffs from +4 to -4.
  for (let i = 0; i <= pointCount; i++) {
    const cutoff = 4 - (8 * i / pointCount);
    fpr.push(1 - normalCDF(cutoff - params.negativeMean));
    tpr.push(1 - normalCDF(cutoff - params.positiveMean));
  }

  return { fpr, tpr };
}

function drawROC() {
  const auc = Number(document.getElementById("aucSlider").value);
  const thresholdControl = Number(
    document.getElementById("thresholdSlider").value
  );

  const params = parametersFromAUC(auc);
  const curve = buildCurve(params);
  const point = calculateRates(thresholdControl, params);

  document.getElementById("aucValue").textContent = auc.toFixed(2);
  document.getElementById("thresholdValue").textContent =
    thresholdControl.toFixed(3);

  const traces = [
    {
      x: curve.fpr,
      y: curve.tpr,
      type: "scatter",
      mode: "lines",
      name: "Theoretical ROC curve",
      line: { width: 3 }
    },
    {
      x: [0, 1],
      y: [0, 1],
      type: "scatter",
      mode: "lines",
      name: "Random classifier",
      line: { dash: "dash" }
    },
    {
      x: [point.fpr],
      y: [point.tpr],
      type: "scatter",
      mode: "markers",
      name: "Selected threshold",
      marker: { size: 14, symbol: "circle" }
    }
  ];

  const layout = {
    title: `Theoretical ROC Curve (AUROC = ${auc.toFixed(2)})`,
    xaxis: {
      title: "False Positive Rate (FPR)",
      range: [0, 1]
    },
    yaxis: {
      title: "True Positive Rate (TPR)",
      range: [0, 1],
      scaleanchor: "x",
      scaleratio: 1
    },
    margin: { t: 65, r: 30, b: 70, l: 75 },
    legend: { orientation: "h", y: -0.22 }
  };

  Plotly.react("rocChart", traces, layout, { responsive: true });

  document.getElementById("metrics").textContent =
    `AUROC: ${auc.toFixed(2)} | ` +
    `Threshold control: ${thresholdControl.toFixed(3)} | ` +
    `Score cutoff: ${point.cutoff.toFixed(3)} | ` +
    `TPR: ${point.tpr.toFixed(3)} | FPR: ${point.fpr.toFixed(3)}`;
}

document.getElementById("aucSlider").addEventListener("input", drawROC);
document.getElementById("thresholdSlider").addEventListener("input", drawROC);

// Initial chart.
drawROC();
