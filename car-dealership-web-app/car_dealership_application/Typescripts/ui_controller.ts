/// <reference types='jquery' />

import $ from 'jquery';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);
import { CHART_COLORS, transparentize } from './color_utils'
let Promise = require('es6-promises');

// --------------------------------------------------------------------------
// Section: Datatable
// --------------------------------------------------------------------------
// Use JQuery to hide and show specific columns based on checkboxes

export class FilterCache {
    private filterCacheName: string;
    private checkboxes: JQuery<HTMLInputElement>;
	private columnCheckBoxes: JQuery<HTMLInputElement>;
	private SELF: FilterCache;

    constructor(filterCacheName: string, checkboxes: JQuery<HTMLInputElement>, columnCheckBoxes: JQuery<HTMLInputElement>) {
        this.filterCacheName = filterCacheName;
        this.checkboxes = checkboxes;
		this.columnCheckBoxes = columnCheckBoxes;
		this.SELF = this;
    }

	private filterColumn(SELF: FilterCache, checkbox: JQuery<HTMLInputElement>): void {
		let filterCacheName = SELF.filterCacheName;
		let checkboxes = SELF.checkboxes;

		let id = checkbox.attr("id");
		let columns: JQuery<HTMLInputElement> = $('.' + id);
		let cachedFilter: string | null = localStorage.getItem(filterCacheName);
		if (typeof (cachedFilter) !== 'string') {
			console.error(`Cache ${filterCacheName} is not found or deleted.`);
		}
		let cacheArr = JSON.parse(cachedFilter as string);
		if (checkbox.prop('checked')) {
			// Update the preference
			if (!cacheArr.includes(id))
				cacheArr.push(id);
			if (cacheArr.length === checkboxes.length)
				$("#selectAll").prop("checked", true);
			SELF.displaySpecificColumn(columns);
		} else {
			// Update the preference
			if (cacheArr.includes(id))
				cacheArr = cacheArr.filter(function (value: string, index: number, arr: Array<string>) { return value != id });
			// Uncheck "Select all" checkbox if users uncheck one of the checkboxes
			$("#selectAll").prop("checked", false);
			SELF.hideSpecificColumn(columns);
		}
		// Save the preference
		localStorage.setItem(SELF.filterCacheName, JSON.stringify(cacheArr));
	}

	private displayAllColumns(SELF: FilterCache, checkboxes: JQuery<HTMLInputElement>): void {
		let filterColumn = SELF.filterColumn;

		checkboxes.each(function () {
			$(this).prop("checked", true);
			filterColumn(SELF, $(this));
		});
	}

	// This function hide all columns "with" updating the cache
	private hideAllColumns(SELF: FilterCache, checkboxes: JQuery<HTMLInputElement>): void {
		let filterColumn = SELF.filterColumn;

		checkboxes.each(function () {
			$(this).prop("checked", false);
			filterColumn(SELF, $(this));
		});
	}

	// This function hide all columns "without" updating the cache
	private hideAllColumns2(SELF: FilterCache, checkboxes: JQuery<HTMLInputElement>): void {
		let hideSpecificColumn = SELF.hideSpecificColumn;

		checkboxes.each(function () {
			let id = $(this).attr("id");
			let columns = $('.' + id) as JQuery<HTMLInputElement>;
			$(this).prop("checked", false);
			hideSpecificColumn(columns);
		});
	}

	private displaySpecificColumn(columns: JQuery<HTMLInputElement>): void {
		$.each(columns, function (index, column) {
			// Show column
			column.style.display = "";
		});
	}

	private hideSpecificColumn(columns: JQuery<HTMLInputElement>): void {
		$.each(columns, function (index, column) {
			// Hide column
			column.style.display = "none";
		});
	}

	loadFilterCache(): void {
		let checkboxes = this.checkboxes;
		let SELF = this.SELF;
		let columnCheckBoxes = this.columnCheckBoxes;
		let displaySpecificColumn = this.displaySpecificColumn;
		let hideAllColumns2 = this.hideAllColumns2;

		if (typeof (Storage) !== "undefined") {
			// If user preference is available in cache
			if (localStorage[SELF.filterCacheName]) {
				// Reset the checkboxes
				$('input[type="checkbox"]').prop('checked', false);
				hideAllColumns2(SELF, checkboxes);
				// Get the user preference from JavaScript Storage API
				let cacheArr = JSON.parse(localStorage[SELF.filterCacheName]);
				/* If the user preference is to display all columns, then
				   display all columns.
				*/
				if (cacheArr.length === columnCheckBoxes.length)
					$("#selectAll").prop('checked', true);

				cacheArr.forEach(function (element: string, index: number) {
					let columns = $('.' + element) as JQuery<HTMLInputElement>;
					$('input[name=filtercolumn]').filter('#' + element).prop('checked', true);
					displaySpecificColumn(columns);
				});
			}
			// If user preference is not available in cache
			else {
				// Display all columns by default and create new cache
				$("#selectAll").prop('checked', true);
				let cacheArr: Array<string> = [];
				columnCheckBoxes.each(function () {
					if ($(this).attr('id') !== undefined) {
						cacheArr.push($(this).attr('id') as string);
						$(this).prop('checked', true);
					}
					else {
						console.warn(`${this} is undefined. Can't push to cache.`);
					}
				});
				localStorage.setItem(SELF.filterCacheName, JSON.stringify(cacheArr));
			}
		}
		else {
			// Check all the checkboxes once the document is loaded if no Web Storage support.
			$('input[type="checkbox"]').prop('checked', true);
		}
	}

	addEventListeners() {
		let checkboxes = this.checkboxes;
		let SELF = this.SELF;
		let displayAllColumns = this.displayAllColumns;
		let hideAllColumns = this.hideAllColumns;
		let filterColumn = this.filterColumn;

		/*
		Once the user click "Select all" checkbox, activate all 
		checkboxes programmatically.
		*/

		$("#selectAll").on("change", function () {
			if ($("#selectAll").prop("checked")) {
				displayAllColumns(SELF, checkboxes);
			} else {
				hideAllColumns(SELF, checkboxes);
			}
		});

		// Activate the filterColumn function on change.	 
		checkboxes.on("change", function () {
			let checkbox = $(this);
			filterColumn(SELF, checkbox);
		});
	}
}

// --------------------------------------------------------------------------
// Section end
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Section: Draggable element functions
// --------------------------------------------------------------------------

interface Position {
    // The current scroll 
    readonly left: number,
    readonly top: number,
    // Get the current mouse position
    readonly x: number,
    readonly y: number,
}

export function enableDrag(className: string): void {
    const elements: JQuery<HTMLElement> = $(`.${className}`);

    elements.each((index: number, element: HTMLElement) => {
        element.scrollTop = 100;
        element.scrollLeft = 150;

        $(element).on('mousedown', mouseDownHandler);
    });
}

// Enable drag when the user trigger the mouse down event
// Source: https://htmldom.dev/drag-to-scroll/
const mouseDownHandler = function (event: JQueryEventObject): void {
    const element: JQuery<Element> = $(event.target);
    
    // Change the cursor and prevent user from selecting the text
    element.css({
        'cursor': 'grabbing',
        'user-select': 'none'
    });

    // Retrieve the positional information of the cursor
    let pos: Position = {
        // The current scroll 
        left: element.prop('scrollLeft'),
        top: element.prop('scrollTop'),
        // Get the current mouse position
        x: event.clientX,
        y: event.clientY,
    };

    // Add the event handler to enable scrolling
    element.on('mousemove', null, pos, mouseMoveHandler);
    // Add the event handler to disable drag
    element.on('mouseup', mouseUpHandler);
};

// Enable scrolling by calculating how far it has been moved
// Source: https://htmldom.dev/drag-to-scroll/
const mouseMoveHandler = function (event: JQueryEventObject): void {
    const element: JQuery<Element> = $(event.target);
    const left = event.data.left;
    const top = event.data.top;
    const x = event.data.x;
    const y = event.data.y;

    // How far the mouse has been moved
    const DX = event.clientX - x;
    const DY = event.clientY - y;

    // Scroll the element
    element.prop('scrollTop', top - DX);
    element.prop('scrollLeft', left - DY);
};

// Disable drag when user releases the mouse down event
// Source: https://htmldom.dev/drag-to-scroll/
const mouseUpHandler = function (event: JQueryEventObject): void {
    const element: JQuery<Element> = $(event.target);

    element.css({
        'cursor': 'grab',
        'user-select': 'auto'
    });
};

// --------------------------------------------------------------------------
// Section end
// --------------------------------------------------------------------------

// Executes every 0.5 seconds to check if json is already loaded 
// from remote API or from service worker (if cached)
// Source: https://stackoverflow.com/a/51212718
export function delay(timer: number): Promise<void> {
    return new Promise((resolve: () => void) => {
        timer = Math.floor(timer) || 500;
        setTimeout(function () {
            resolve();
        }, timer);
    });
};

// Hide loading screen and show the content
export function setVisible(selector: string, visible: boolean): void {
    let elem = $(selector).css('display');
    if (visible) {
        $(selector).addClass('display-block-important');
    } else {
        $(selector).addClass('display-none-important');
    }
}

// --------------------------------------------------------------------------
// Section: Chart.js
// --------------------------------------------------------------------------

// Specify custom type for feature importance
export type FeatureImportance = {
	[key: string]: number
}

export function preprocessFIs(feature_importances: Array<FeatureImportance>): [string[], number[], string[], number[]] {
	let [raw_fi, engineered_fi] = feature_importances;
	// Convert to json
	let raw_fi_json = [];
	for (let feature of Object.keys(raw_fi)) {
		raw_fi_json.push({ "feature": feature, "value": raw_fi[feature] });
	}
	let engineered_fi_json = [];
	for (let feature of Object.keys(engineered_fi)) {
		engineered_fi_json.push({ "feature": feature, "value": engineered_fi[feature] });
	}
	// Sort
	let raw_fi_desc = raw_fi_json.sort(function (a, b) {
		return b.value - a.value;
	});
	let engineered_fi_desc = engineered_fi_json.sort(function (a, b) {
		return b.value - a.value;
	});

	// Remap variables
	let rawFi = raw_fi_desc;
	let engineeredFi = engineered_fi_desc;

	// Preprocess each feature importances into x and y
	rawFi = rawFi.filter(elem => elem['value'] > 0);
	let rawFiX = rawFi.map((item) => item['feature']);
	let rawFiY = rawFi.map((item) => item['value']);

	engineeredFi = engineeredFi.filter(elem => elem['value'] > 0);
	let engineeredFiX = engineeredFi.map((item) => item['feature']);
	let engineeredFiY = engineeredFi.map((item) => item['value']);

	return [rawFiX, rawFiY, engineeredFiX, engineeredFiY]
}

export interface BarChartConfig {
	DOM: HTMLCanvasElement,
	x: Array<string>,
	y: Array<number>,
	xLabel: string,
	yLabel: string,
	legendLabel: string,
	borderColor: CHART_COLORS,
	backgroundColor: string,
	borderWidth: 1
};

// Create chart.js chart
export function createBarChart(config: BarChartConfig): Chart {
	return new Chart(config['DOM'], {
		type: 'bar',
		data: {
			labels: config['x'],
			datasets: [{
				label: config['legendLabel'],
				data: config['y'],
				borderColor: config['borderColor'],
				backgroundColor: config['backgroundColor'],
				borderWidth: config['borderWidth']
			}]
		},
		options: {
			responsive: true,
			plugins: {
				title: {
					display: false,
					//text: 'Suggested Min and Max Settings'
				},
				//legend: {
				//    position: 'right'
				//}
				zoom: {
					pan: {
						enabled: true,
						mode: 'xy',
						threshold: 5,
					},
					zoom: {
						wheel: {
							enabled: true
						},
						pinch: {
							enabled: true
						},
						mode: 'xy',
					},
				},
			},
			scales: {
				x: {
					display: true,
					title: {
						display: true,
						text: config['xLabel'],
					}
				},
				y: {
					display: true,
					title: {
						display: true,
						text: config['yLabel'],
						//color: '#191',
						//font: {
						//    family: 'Times',
						//    size: 20,
						//    style: 'normal',
						//    lineHeight: 1.2
						//},
						//padding: { top: 30, left: 0, right: 0, bottom: 0 }
					},
					beginAtZero: false,
					// suggestedMin: 30,
					// suggestedMax: 50,
				}
			}
		},
	});
}

// --------------------------------------------------------------------------
// Section end
// --------------------------------------------------------------------------