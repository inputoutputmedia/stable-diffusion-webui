function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }

    return elem.shadowRoot ? elem.shadowRoot : elem;
}

var decentre = false;

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {

    var elem = gradioApp().querySelector('#tabs > .tab-nav > button.selected');
    var txt = elem.innerText;
    if (txt == "Decentre") {
        decentre = true;
    }
    else {
        decentre = false;
    }

    setTimeout(dld_click1, 500);

    return elem;
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {

    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
    
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#txt2img_prompt')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

/**
 * Add keyboard shortcuts:
 * Ctrl+Enter to start/restart a generation
 * Alt/Option+Enter to skip a generation
 * Esc to interrupt a generation
 */
document.addEventListener('keydown', function(e) {
    const isEnter = e.key === 'Enter' || e.keyCode === 13;
    const isCtrlKey = e.metaKey || e.ctrlKey;
    const isAltKey = e.altKey;
    const isEsc = e.key === 'Escape';

    const generateButton = get_uiCurrentTabContent().querySelector('button[id$=_generate]');
    const interruptButton = get_uiCurrentTabContent().querySelector('button[id$=_interrupt]');
    const skipButton = get_uiCurrentTabContent().querySelector('button[id$=_skip]');

    if (isCtrlKey && isEnter) {
        if (interruptButton.style.display === 'block') {
            interruptButton.click();
            const callback = (mutationList) => {
                for (const mutation of mutationList) {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        if (interruptButton.style.display === 'none') {
                            generateButton.click();
                            observer.disconnect();
                        }
                    }
                }
            };
            const observer = new MutationObserver(callback);
            observer.observe(interruptButton, {attributes: true});
        } else {
            generateButton.click();
        }
        e.preventDefault();
    }

    if (isAltKey && isEnter) {
        skipButton.click();
        e.preventDefault();
    }

    if (isEsc) {
        const globalPopup = document.querySelector('.global-popup');
        const lightboxModal = document.querySelector('#lightboxModal');
        if (!globalPopup || globalPopup.style.display === 'none') {
            if (document.activeElement === lightboxModal) return;
            if (interruptButton.style.display === 'block') {
                interruptButton.click();
                e.preventDefault();
            }
        }
    }
});

function shortcuts(e) {

    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        case "select":
        case "button":
        break;
        default:
        if (e.code == "KeyA") {
            document.getElementById("buttonAdd").click();
        }
        if (e.code == "KeyD") {
            document.getElementById("buttonRemove").click();
        }
        if(e.code == "ArrowLeft") {
          document.getElementById("buttonLeft").click();
        }
        if(e.code == "ArrowRight") {
          document.getElementById("buttonRight").click();
        }
}
}


document.addEventListener('keyup', shortcuts, false);

function AddToDataBase(){
    document.getElementById("detect").click();
       document.getElementById("caption").click();
}

function AddMultiToDataBase(){
    document.getElementById("multiDetect").click();
       document.getElementById("multiCaption").click();
}

function dld_click1() {
    document.getElementById("dldbtn").click();
}

var clkdld = false;

function dld_click() {
    if(decentre) {
     if(!clkdld && confirm('Do you want to re download the caption models?') == true) {
        clkdld = true;
        return true;
      } else {
        clkdld = true;
        return false;
      }
    }
    clkdld = false;
    return false;
}

function disable_buttons(){
    document.getElementById("buttonAdd").disabled = true;
    document.getElementById("buttonLeft").disabled = true;
    document.getElementById("buttonRight").disabled = true;
    document.getElementById("buttonRemove").disabled = true;
    document.getElementById("addToDB").disabled = true;
    document.getElementById("addAIToDB").disabled = true;
    document.getElementById("importF").disabled = true;
    document.getElementById("importU").disabled = true;
    document.getElementById("reset1").disabled = true;
    document.getElementById("reset2").disabled = true;
    document.getElementById("save1").disabled = true;
    document.getElementById("save2").disabled = true;
    document.getElementById("processF").disabled = true;
}

function enabled_buttons(){
    document.getElementById("buttonAdd").disabled = false;
    document.getElementById("buttonLeft").disabled = false;
    document.getElementById("buttonRight").disabled = false;
    document.getElementById("buttonRemove").disabled = false;
    document.getElementById("addToDB").disabled = false;
    document.getElementById("addAIToDB").disabled = false;
    document.getElementById("importF").disabled = false;
    document.getElementById("importU").disabled = false;
    document.getElementById("reset1").disabled = false;
    document.getElementById("reset2").disabled = false;
    document.getElementById("save1").disabled = false;
    document.getElementById("save2").disabled = false;
    document.getElementById("processF").disabled = false;
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}
