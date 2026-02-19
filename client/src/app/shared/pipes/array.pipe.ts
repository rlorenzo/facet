import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'includes', standalone: true, pure: true })
export class IncludesPipe implements PipeTransform {
  transform(array: readonly unknown[], item: unknown): boolean {
    return array.includes(item);
  }
}

@Pipe({ name: 'indexOf', standalone: true, pure: true })
export class IndexOfPipe implements PipeTransform {
  transform(array: readonly unknown[], item: unknown): number {
    return array.indexOf(item);
  }
}
