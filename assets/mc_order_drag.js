(function () {
  function updateStore(list) {
    if (!list || !window.dash_clientside || !window.dash_clientside.set_props) {
      return;
    }
    const values = Array.from(list.querySelectorAll(".mc-order-item")).map(
      (item) => item.dataset.value
    );
    window.dash_clientside.set_props("mc-order-store", { data: values });
  }

  function attach(list) {
    if (!list || list.dataset.dragInit === "true") {
      return;
    }
    list.dataset.dragInit = "true";
    let dragItem = null;

    list.addEventListener("dragstart", (event) => {
      const item = event.target.closest(".mc-order-item");
      if (!item) {
        return;
      }
      dragItem = item;
      dragItem.classList.add("mc-order-dragging");
      event.dataTransfer.effectAllowed = "move";
    });

    list.addEventListener("dragend", () => {
      if (dragItem) {
        dragItem.classList.remove("mc-order-dragging");
        dragItem = null;
      }
      updateStore(list);
    });

    list.addEventListener("dragover", (event) => {
      if (!dragItem) {
        return;
      }
      event.preventDefault();
      const target = event.target.closest(".mc-order-item");
      if (!target || target === dragItem) {
        return;
      }
      const rect = target.getBoundingClientRect();
      const next = event.clientY - rect.top > rect.height / 2;
      list.insertBefore(dragItem, next ? target.nextSibling : target);
    });

    list.addEventListener("drop", (event) => {
      event.preventDefault();
      updateStore(list);
    });
  }

  function init() {
    const list = document.getElementById("mc-order-list");
    if (list) {
      attach(list);
    }
  }

  const observer = new MutationObserver(init);
  observer.observe(document.body, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", init);
})();
