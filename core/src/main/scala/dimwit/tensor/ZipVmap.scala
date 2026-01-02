package dimwit.tensor

import dimwit.jax.Jax
import dimwit.tensor.TupleHelpers.Remover
import dimwit.tensor.TensorOps.Structural.slice
import me.shadaj.scalapy.py.SeqConverters

/** Provides zipvmap operations for efficiently mapping functions over multiple tensors along a common axis.
  *
  * This module includes optimized JAX-native implementations for 2, 3, and 4 tensor cases, with a fallback to Scala-side iteration for other arities.
  */
object ZipVmap:

  // -----------------------------------------------------------
  // Type-level helpers for working with tuples of tensors
  // -----------------------------------------------------------

  /** Reconstructs a tuple of tensors from separate shape and value tuples */
  type TensorsOf[Shapes <: Tuple, Values <: Tuple] <: Tuple = (Shapes, Values) match
    case (EmptyTuple, EmptyTuple)                             => EmptyTuple
    case ((shapeHead *: shapeTail), (valueHead *: valueTail)) => Tensor[shapeHead, valueHead] *: TensorsOf[shapeTail, valueTail]

  /** Extracts the shape type from a Tensor type */
  type ExtractShape[T] = T match
    case Tensor[s, v] => s

  /** Extracts the value type from a Tensor type */
  type ExtractValue[T] = T match
    case Tensor[s, v] => v

  /** Maps a tuple of tensors to their shape types */
  type ShapesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractShape]

  /** Maps a tuple of tensors to their value types */
  type ValuesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractValue]

  // -----------------------------------------------------------
  // Zipper typeclass for slicing tuples of tensors
  // -----------------------------------------------------------

  /** Typeclass for operations on tuples of tensors that share a common axis. Provides the ability to slice all tensors in a tuple along that axis.
    */
  trait Zipper[Shapes <: Tuple, L, Values <: Tuple]:
    type SlicedShapes <: Tuple
    def dimSize(tensors: TensorsOf[Shapes, Values], axis: Axis[L]): Int
    def sliceAll(tensors: TensorsOf[Shapes, Values], axis: Axis[L], idx: Int): TensorsOf[SlicedShapes, Values]

  object Zipper:
    type Aux[Shapes <: Tuple, L, Values <: Tuple, O <: Tuple] = Zipper[Shapes, L, Values] { type SlicedShapes = O }

    given empty[L]: Zipper.Aux[EmptyTuple, L, EmptyTuple, EmptyTuple] = new Zipper[EmptyTuple, L, EmptyTuple]:
      type SlicedShapes = EmptyTuple
      def dimSize(t: EmptyTuple, axis: Axis[L]) = 0
      def sliceAll(t: EmptyTuple, axis: Axis[L], idx: Int) = EmptyTuple

    given cons[HeadShape <: Tuple: Labels, TailShapes <: Tuple, L: Label, TailSliced <: Tuple, HeadValue, TailValues <: Tuple, R <: Tuple](using
        remover: Remover.Aux[HeadShape, L, R],
        axisIndex: AxisIndex[HeadShape, L],
        tailZipper: Zipper.Aux[TailShapes, L, TailValues, TailSliced],
        labels: Labels[R]
    ): Zipper.Aux[HeadShape *: TailShapes, L, HeadValue *: TailValues, R *: TailSliced] =
      new Zipper[HeadShape *: TailShapes, L, HeadValue *: TailValues]:
        type SlicedShapes = R *: TailSliced

        def dimSize(tensors: TensorsOf[HeadShape *: TailShapes, HeadValue *: TailValues], axis: Axis[L]): Int =
          val head = tensors.asInstanceOf[Tensor[HeadShape, HeadValue] *: Tuple].head
          head.shape.dimensions(axisIndex.value)

        def sliceAll(tensors: TensorsOf[HeadShape *: TailShapes, HeadValue *: TailValues], axis: Axis[L], idx: Int): TensorsOf[SlicedShapes, HeadValue *: TailValues] =
          val tuple = tensors.asInstanceOf[Tensor[HeadShape, HeadValue] *: TensorsOf[TailShapes, TailValues]]
          val slicedHead = tuple.head.slice(axis -> idx)
          val slicedTail = tailZipper.sliceAll(tuple.tail, axis, idx)
          (slicedHead *: slicedTail).asInstanceOf[TensorsOf[SlicedShapes, HeadValue *: TailValues]]

  // -----------------------------------------------------------
  // ZipResult - deferred vmap computation
  // -----------------------------------------------------------

  /** Represents a zipped tuple of tensors along a common axis. Provides a vmap method for Scala-side iteration (used as fallback).
    */
  case class ZipResult[L: Label, Shapes <: Tuple, Values <: Tuple](
      axis: Axis[L],
      tensors: TensorsOf[Shapes, Values]
  ):
    def vmap[OutShape <: Tuple: Labels, OutV](using
        zipper: Zipper[Shapes, L, Values]
    )(
        f: TensorsOf[zipper.SlicedShapes, Values] => Tensor[OutShape, OutV]
    ): Tensor[L *: OutShape, OutV] =

      val size = zipper.dimSize(tensors, axis)

      val results = (0 until size).map { i =>
        val slicedTuple = zipper.sliceAll(tensors, axis, i)
        f(slicedTuple)
      }

      TensorOps.Structural.stack(results, axis)

  // -----------------------------------------------------------
  // ZipVmapImpl - dispatcher trait for optimized implementations
  // -----------------------------------------------------------

  /** Typeclass for dispatching zipvmap to the most efficient implementation based on the number of tensors.
    */
  trait ZipVmapImpl[Shapes <: Tuple, L, Values <: Tuple, SlicedShapes <: Tuple]:
    def apply[OutShape <: Tuple: Labels, OutV](
        axis: Axis[L],
        tensors: TensorsOf[Shapes, Values],
        f: TensorsOf[SlicedShapes, Values] => Tensor[OutShape, OutV]
    ): Tensor[L *: OutShape, OutV]

  object ZipVmapImpl:

    // High-priority: 2-tensor JAX-native implementation
    given twoTensors[L: Label, T1 <: Tuple: Labels, T2 <: Tuple: Labels, V1, V2, R1 <: Tuple, R2 <: Tuple](using
        remover1: Remover.Aux[T1, L, R1],
        remover2: Remover.Aux[T2, L, R2],
        axisIndex1: AxisIndex[T1, L],
        axisIndex2: AxisIndex[T2, L],
        labels1: Labels[R1],
        labels2: Labels[R2]
    ): ZipVmapImpl[T1 *: T2 *: EmptyTuple, L, V1 *: V2 *: EmptyTuple, R1 *: R2 *: EmptyTuple] with
      def apply[OutShape <: Tuple: Labels, OutV](
          axis: Axis[L],
          tensors: TensorsOf[T1 *: T2 *: EmptyTuple, V1 *: V2 *: EmptyTuple],
          f: TensorsOf[R1 *: R2 *: EmptyTuple, V1 *: V2 *: EmptyTuple] => Tensor[OutShape, OutV]
      ): Tensor[L *: OutShape, OutV] =
        val tuple = tensors.asInstanceOf[Tensor[T1, V1] *: Tensor[T2, V2] *: EmptyTuple]
        val t1 = tuple.head
        val t2 = tuple.tail.head

        val fpy = (jxpr1: Jax.PyDynamic, jxpr2: Jax.PyDynamic) =>
          val innerTensor1 = Tensor[R1, V1](jxpr1)
          val innerTensor2 = Tensor[R2, V2](jxpr2)
          val innerTuple = (innerTensor1 *: innerTensor2 *: EmptyTuple).asInstanceOf[TensorsOf[R1 *: R2 *: EmptyTuple, V1 *: V2 *: EmptyTuple]]
          val result = f(innerTuple)
          result.jaxValue

        val vmapAxes = Jax.Dynamic.global.tuple(Seq(axisIndex1.value, axisIndex2.value).toPythonProxy)
        Tensor(Jax.jax_helper.vmapN(fpy, vmapAxes)(t1.jaxValue, t2.jaxValue))

    // High-priority: 3-tensor JAX-native implementation
    given threeTensors[L: Label, T1 <: Tuple: Labels, T2 <: Tuple: Labels, T3 <: Tuple: Labels, V1, V2, V3, R1 <: Tuple, R2 <: Tuple, R3 <: Tuple](using
        remover1: Remover.Aux[T1, L, R1],
        remover2: Remover.Aux[T2, L, R2],
        remover3: Remover.Aux[T3, L, R3],
        axisIndex1: AxisIndex[T1, L],
        axisIndex2: AxisIndex[T2, L],
        axisIndex3: AxisIndex[T3, L],
        labels1: Labels[R1],
        labels2: Labels[R2],
        labels3: Labels[R3]
    ): ZipVmapImpl[T1 *: T2 *: T3 *: EmptyTuple, L, V1 *: V2 *: V3 *: EmptyTuple, R1 *: R2 *: R3 *: EmptyTuple] with
      def apply[OutShape <: Tuple: Labels, OutV](
          axis: Axis[L],
          tensors: TensorsOf[T1 *: T2 *: T3 *: EmptyTuple, V1 *: V2 *: V3 *: EmptyTuple],
          f: TensorsOf[R1 *: R2 *: R3 *: EmptyTuple, V1 *: V2 *: V3 *: EmptyTuple] => Tensor[OutShape, OutV]
      ): Tensor[L *: OutShape, OutV] =
        val tuple = tensors.asInstanceOf[Tensor[T1, V1] *: Tensor[T2, V2] *: Tensor[T3, V3] *: EmptyTuple]
        val t1 = tuple.head
        val t2 = tuple.tail.head
        val t3 = tuple.tail.tail.head

        val fpy = (jxpr1: Jax.PyDynamic, jxpr2: Jax.PyDynamic, jxpr3: Jax.PyDynamic) =>
          val innerTensor1 = Tensor[R1, V1](jxpr1)
          val innerTensor2 = Tensor[R2, V2](jxpr2)
          val innerTensor3 = Tensor[R3, V3](jxpr3)
          val innerTuple = (innerTensor1 *: innerTensor2 *: innerTensor3 *: EmptyTuple).asInstanceOf[TensorsOf[R1 *: R2 *: R3 *: EmptyTuple, V1 *: V2 *: V3 *: EmptyTuple]]
          val result = f(innerTuple)
          result.jaxValue

        val vmapAxes = Jax.Dynamic.global.tuple(Seq(axisIndex1.value, axisIndex2.value, axisIndex3.value).toPythonProxy)
        Tensor(Jax.jax_helper.vmapN(fpy, vmapAxes)(t1.jaxValue, t2.jaxValue, t3.jaxValue))

    // High-priority: 4-tensor JAX-native implementation
    given fourTensors[L: Label, T1 <: Tuple: Labels, T2 <: Tuple: Labels, T3 <: Tuple: Labels, T4 <: Tuple: Labels, V1, V2, V3, V4, R1 <: Tuple, R2 <: Tuple, R3 <: Tuple, R4 <: Tuple](using
        remover1: Remover.Aux[T1, L, R1],
        remover2: Remover.Aux[T2, L, R2],
        remover3: Remover.Aux[T3, L, R3],
        remover4: Remover.Aux[T4, L, R4],
        axisIndex1: AxisIndex[T1, L],
        axisIndex2: AxisIndex[T2, L],
        axisIndex3: AxisIndex[T3, L],
        axisIndex4: AxisIndex[T4, L],
        labels1: Labels[R1],
        labels2: Labels[R2],
        labels3: Labels[R3],
        labels4: Labels[R4]
    ): ZipVmapImpl[T1 *: T2 *: T3 *: T4 *: EmptyTuple, L, V1 *: V2 *: V3 *: V4 *: EmptyTuple, R1 *: R2 *: R3 *: R4 *: EmptyTuple] with
      def apply[OutShape <: Tuple: Labels, OutV](
          axis: Axis[L],
          tensors: TensorsOf[T1 *: T2 *: T3 *: T4 *: EmptyTuple, V1 *: V2 *: V3 *: V4 *: EmptyTuple],
          f: TensorsOf[R1 *: R2 *: R3 *: R4 *: EmptyTuple, V1 *: V2 *: V3 *: V4 *: EmptyTuple] => Tensor[OutShape, OutV]
      ): Tensor[L *: OutShape, OutV] =
        val tuple = tensors.asInstanceOf[Tensor[T1, V1] *: Tensor[T2, V2] *: Tensor[T3, V3] *: Tensor[T4, V4] *: EmptyTuple]
        val t1 = tuple.head
        val t2 = tuple.tail.head
        val t3 = tuple.tail.tail.head
        val t4 = tuple.tail.tail.tail.head

        val fpy = (jxpr1: Jax.PyDynamic, jxpr2: Jax.PyDynamic, jxpr3: Jax.PyDynamic, jxpr4: Jax.PyDynamic) =>
          val innerTensor1 = Tensor[R1, V1](jxpr1)
          val innerTensor2 = Tensor[R2, V2](jxpr2)
          val innerTensor3 = Tensor[R3, V3](jxpr3)
          val innerTensor4 = Tensor[R4, V4](jxpr4)
          val innerTuple = (innerTensor1 *: innerTensor2 *: innerTensor3 *: innerTensor4 *: EmptyTuple).asInstanceOf[TensorsOf[R1 *: R2 *: R3 *: R4 *: EmptyTuple, V1 *: V2 *: V3 *: V4 *: EmptyTuple]]
          val result = f(innerTuple)
          result.jaxValue

        val vmapAxes = Jax.Dynamic.global.tuple(Seq(axisIndex1.value, axisIndex2.value, axisIndex3.value, axisIndex4.value).toPythonProxy)
        Tensor(Jax.jax_helper.vmapN(fpy, vmapAxes)(t1.jaxValue, t2.jaxValue, t3.jaxValue, t4.jaxValue))

    // Low-priority fallback: use Scala-side iteration for N tensors (N != 2, 3, 4)
    given fallback[Shapes <: Tuple, L: Label, Values <: Tuple, SlicedShapes <: Tuple](using
        zipper: Zipper.Aux[Shapes, L, Values, SlicedShapes]
    ): ZipVmapImpl[Shapes, L, Values, SlicedShapes] with
      def apply[OutShape <: Tuple: Labels, OutV](
          axis: Axis[L],
          tensors: TensorsOf[Shapes, Values],
          f: TensorsOf[SlicedShapes, Values] => Tensor[OutShape, OutV]
      ): Tensor[L *: OutShape, OutV] =
        val size = zipper.dimSize(tensors, axis)
        val results = (0 until size).map { i =>
          val slicedTuple = zipper.sliceAll(tensors, axis, i)
          f(slicedTuple)
        }
        TensorOps.Structural.stack(results, axis)

  // -----------------------------------------------------------
  // Public API
  // -----------------------------------------------------------

  /** Zips multiple tensors along a common axis without immediately applying a function. Returns a ZipResult that can be used with the vmap method.
    */
  def zip[L: Label, Inputs <: Tuple](
      axis: Axis[L]
  )(
      tensors: Inputs
  ): ZipResult[L, ShapesOf[Inputs], ValuesOf[Inputs]] =
    ZipResult(axis, tensors.asInstanceOf[TensorsOf[ShapesOf[Inputs], ValuesOf[Inputs]]])

  /** Maps a function over multiple tensors along a common axis.
    *
    * Automatically selects the most efficient implementation:
    *   - 2 tensors: JAX-native vmap2 (single boundary crossing)
    *   - 3 tensors: JAX-native vmap3 (single boundary crossing)
    *   - 4 tensors: JAX-native vmap4 (single boundary crossing)
    *   - Other: Scala-side iteration with manual stacking
    *
    * @param axis
    *   The axis along which to map
    * @param tensors
    *   A tuple of tensors with the specified axis
    * @param f
    *   The function to apply to slices of the tensors
    */
  def zipvmap[
      L: Label,
      Inputs <: Tuple,
      OutShape <: Tuple: Labels,
      OutV
  ](
      axis: Axis[L]
  )(
      tensors: Inputs
  )(using
      zipper: Zipper[ShapesOf[Inputs], L, ValuesOf[Inputs]],
      impl: ZipVmapImpl[ShapesOf[Inputs], L, ValuesOf[Inputs], zipper.SlicedShapes]
  )(
      f: TensorsOf[zipper.SlicedShapes, ValuesOf[Inputs]] => Tensor[OutShape, OutV]
  ): Tensor[L *: OutShape, OutV] =
    impl.apply(axis, tensors.asInstanceOf[TensorsOf[ShapesOf[Inputs], ValuesOf[Inputs]]], f)

end ZipVmap
