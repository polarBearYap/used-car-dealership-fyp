export enum CHART_COLORS {
    red = 'rgb(255, 99, 132)',
    orange = 'rgb(255, 159, 64)',
    yellow = 'rgb(255, 205, 86)',
    green = 'rgb(75, 192, 192)',
    blue = 'rgb(54, 162, 235)',
    purple = 'rgb(153, 102, 255)',
    grey = 'rgb(201, 203, 207)'
};

export function transparentize(rgb_value: string, opacity: number): string {
    const colorRegex = /rgb\(\d+\s*,\s*\d+,\s*\d+\)/
    if (colorRegex.test(rgb_value)) {
        if (opacity != NaN) {
            return `rgba${rgb_value.slice(3, -1)}, ${opacity})`;
        }
        else {
            throw 'Invalid opacity value. The opacity value must be a float value.';
        }
    }
    else {
        throw 'Invalid rgb value. rgb must be in the format of rgb(255,255,255).';
    }
}